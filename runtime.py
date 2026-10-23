# 在进程内适配现有 Runner，不修改 AstrBot 本体源码。

from __future__ import annotations

import ast
import inspect
import json
import logging
import time
from contextlib import aclosing
from functools import wraps

from astrbot.api import logger
from astrbot.core.agent.runners.base import AgentState
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.provider.entities import LLMResponse

TOOL_NAME = "deep_letter"
SILENT_ACTIONS = {"send", "broadcast", "reply", "save", "rename", "delete"}
REQUIRED_METHODS = (
    "step",
    "done",
    "_transition_state",
    "_resolve_unconsumed_follow_ups",
    "_is_stop_requested",
    "was_aborted",
)


class LetterToolLogFilter(logging.Filter):
    # 过滤常规信箱工具日志，保留诊断信息及调用追踪。

    def filter(self, record: logging.LogRecord) -> bool:
        # 判断当前日志是否应继续显示。
        if record.levelno != logging.INFO or record.module != "tool_loop_agent_runner":
            return True
        message = record.getMessage()
        if message.startswith("使用工具：deep_letter，参数："):
            try:
                args = ast.literal_eval(message.split("，参数：", 1)[1])
            except (ValueError, SyntaxError):
                return True
            if not isinstance(args, dict):
                return True
            return False
        if message.startswith("Tool `deep_letter` Result:"):
            return False
        if message.startswith("Agent 使用工具: "):
            # 混合调用保留日志，避免连其他工具的诊断信息一起隐藏。
            try:
                names = ast.literal_eval(message.split(": ", 1)[1])
            except (ValueError, SyntaxError):
                return True
            return not (
                isinstance(names, list)
                and names
                and all(name == TOOL_NAME for name in names)
            )
        return True


class LetterRuntime:
    # 管理可撤销的进程内 Runner 适配。

    def __init__(self, owner):
        self.owner = owner
        self.active = False
        self.original = None
        self.wrapper = None
        self.disabled_reason = ""
        self.log_filter = LetterToolLogFilter()

    def disable(self, reason: str) -> None:
        # 停用适配并回退原生流程，不修改用户配置中的隐藏开关。
        self.uninstall()
        if self.disabled_reason != reason:
            logger.warning("信箱隐藏模式已停用，恢复原生流程：%s", reason)
        self.disabled_reason = reason

    def check(self) -> None:
        # 检查接口和包装是否改变，必要时停用适配，不重跑工具。
        if not self.owner.config.get("hide_enabled", True):
            self.uninstall()
            self.disabled_reason = "配置关闭"
        elif self.active and (
            ToolLoopAgentRunner.step is not self.wrapper
            or not all(
                callable(getattr(ToolLoopAgentRunner, name, None))
                for name in REQUIRED_METHODS
            )
        ):
            self.disable("Runner 接口或包装已被其他代码更改")

    def install(self) -> None:
        # 安装运行适配，发现已知接口不兼容时回退原生流程。
        if self.disabled_reason == "配置关闭" and self.owner.config.get(
            "hide_enabled", True
        ):
            self.disabled_reason = ""
        self.check()
        if not self.owner.config.get("hide_enabled", True) or self.disabled_reason:
            return
        if self.active:
            return
        cls = ToolLoopAgentRunner
        if not all(callable(getattr(cls, name, None)) for name in REQUIRED_METHODS):
            self.disable("AstrBot Runner 接口不兼容")
            return
        if not inspect.isasyncgenfunction(cls.step):
            self.disable("Runner.step 不再是异步生成器")
            return
        if getattr(cls.step, "_deep_letter_adapter", None) is not None:
            self.disable("已有其他信箱 Runner 适配")
            return
        original = cls.step
        adapter = self

        @wraps(original)
        async def wrapped(runner):
            adapter.check()
            try:
                context = getattr(getattr(runner, "run_context", None), "context", None)
                req = getattr(runner, "req", None)
                tools = getattr(runner, "_skill_like_raw_tool_set", None)
                if tools is None:
                    tools = getattr(req, "func_tool", None)
                tool = tools.get_tool(TOOL_NAME) if tools is not None else None
                enabled = (
                    adapter.active
                    and context is not None
                    and getattr(context, "context", None) is adapter.owner.context
                    and tool is not None
                    and tool.handler_module_path == type(adapter.owner).__module__
                )
            except Exception:
                adapter.disable("读取 Runner 工具状态失败")
                enabled = False
            if not enabled:
                async with aclosing(original(runner)) as responses:
                    async for response in responses:
                        yield response
                return

            # 原 Runner 负责执行工具和写入上下文，这里只筛选信箱的界面提示。
            event = context.event
            calls = []
            letter_ids = set()
            results = {}
            async with aclosing(original(runner)) as responses:
                async for response in responses:
                    if not adapter.active:
                        yield response
                        continue
                    try:
                        if response.type in {"tool_call", "tool_call_result"}:
                            chain = response.data["chain"]
                            data = (
                                getattr(chain.chain[0], "data", None)
                                if chain.chain
                                else None
                            )
                            if isinstance(data, dict):
                                call_id = data.get("id")
                                if response.type == "tool_call":
                                    calls.append(data)
                                    if data.get("name") == TOOL_NAME:
                                        letter_ids.add(call_id)
                                        # 隐藏聊天界面提示，但保留工具调用追踪。
                                        event.trace.record(
                                            "agent_tool_call", tool_name=data
                                        )
                                        continue
                                elif call_id in letter_ids:
                                    results.setdefault(call_id, []).append(
                                        data.get("result")
                                    )
                                    event.trace.record(
                                        "agent_tool_result",
                                        tool_result=chain.get_plain_text(
                                            with_other_comps_mark=True
                                        ),
                                    )
                                    continue
                    except Exception:
                        # 适配异常时沿原生成器继续，不能重跑已执行的工具。
                        adapter.disable("解析或记录工具提示时发生适配异常")
                    yield response

            try:
                # 到这里原 Runner 已把工具调用与结果成对写入上下文。
                if (
                    not adapter.active
                    or len(calls) != 1
                    or calls[0].get("name") != TOOL_NAME
                    or runner.done()
                    or runner.was_aborted()
                    or runner._is_stop_requested()
                    or event.is_stopped()
                    # 已处理过的追加消息也计入；有追加消息时保持正常续聊。
                    or getattr(runner, "_follow_up_seq", None) != 0
                ):
                    return
                args = calls[0].get("args")
                action = (
                    str(args.get("action", "")).strip().lower()
                    if isinstance(args, dict)
                    else ""
                )
                values = results.get(calls[0].get("id"), [])
                wants_silent = action == "silent" or (
                    action in SILENT_ACTIONS and args.get("silent", True) is True
                )
                if not wants_silent or len(values) != 1:
                    return
                try:
                    # 只解析开头的结果 JSON，兼容 AstrBot 在后面附加限制提示。
                    result, _ = json.JSONDecoder().raw_decode(values[0])
                except (TypeError, ValueError):
                    return
                if (
                    not isinstance(result, dict)
                    or result.get("ok") is not True
                    or result.get("action") != action
                ):
                    return
                if not req.tool_calls_result:
                    return

                # 符合条件才直接结束，不再请求模型；此前输出的正文照常保留。
                final = LLMResponse(role="assistant", completion_text="")
                runner.final_llm_resp = final
                runner._transition_state(AgentState.DONE)
                runner.stats.end_time = time.time()
                try:
                    await runner.agent_hooks.on_agent_done(runner.run_context, final)
                except Exception:
                    logger.exception("信箱调用会话结束钩子失败")
                finally:
                    runner._resolve_unconsumed_follow_ups()

            except Exception:
                adapter.disable("结束 Runner 时发生适配异常")

        wrapped._deep_letter_adapter = self
        self.original = original
        self.wrapper = wrapped
        self.active = True
        cls.step = wrapped
        logging.getLogger("astrbot").addFilter(self.log_filter)

    def uninstall(self) -> None:
        # 停用当前适配；若仍由本插件直接包装，则恢复原方法。
        self.active = False
        logging.getLogger("astrbot").removeFilter(self.log_filter)
        if self.wrapper is not None and ToolLoopAgentRunner.step is self.wrapper:
            ToolLoopAgentRunner.step = self.original
