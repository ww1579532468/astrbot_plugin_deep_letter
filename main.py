from __future__ import annotations

import asyncio
import json
import re
import stat
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from astrbot.api import llm_tool, logger
from astrbot.api.event import MessageChain, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import At, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, Group, MessageMember
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.workspace import resolve_workspace_root_for_umo

from .runtime import LetterRuntime

PLUGIN_NAME = "astrbot_plugin_deep_letter"
MAX_ROUNDS = 10
SEND_COOLDOWN_SECONDS = 60
LETTER_TTL_SECONDS = 24 * 60 * 60
BAD_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
ACTIONS = {
    "send",
    "broadcast",
    "reply",
    "save",
    "rename",
    "delete",
    "read",
    "list",
    "silent",
}


@dataclass(frozen=True)
class Letter:
    letter_id: str
    source: str
    target: str
    content: str
    round_no: int = 1
    is_broadcast: bool = False


class InternalLetterEvent(AstrMessageEvent):
    def __init__(
        self,
        context: Context,
        session: MessageSession,
        platform_meta: PlatformMetadata,
        text: str,
        source_name: str,
        target_name: str,
    ) -> None:
        # 构造内部来信，后续唤醒、模型处理和上下文记录交给 AstrBot 正常管道。
        message = AstrBotMessage()
        message.type = session.message_type
        message.self_id = "letter"
        message.session_id = session.session_id
        if session.message_type == MessageType.GROUP_MESSAGE:
            message.group = Group(group_id=session.session_id, group_name=target_name)
        message.message_id = uuid.uuid4().hex
        message.sender = MessageMember(user_id="astrbot", nickname=source_name)
        message.message = [At(qq="letter"), Plain(text)]
        message.message_str = text
        message.raw_message = text
        message.timestamp = int(time.time())
        super().__init__(text, message, platform_meta, session.session_id)
        self.context_obj = context
        self.is_wake = True
        self.is_at_or_wake_command = True

    async def send(self, message: MessageChain) -> None:
        # 通过目标平台的会话发送接口输出普通回复。
        if message is None or not message.chain:
            return
        if not await self.context_obj.send_message(self.session, message):
            raise RuntimeError("目标平台未运行")
        await super().send(message)

    async def send_streaming(self, generator, use_fallback: bool = False) -> None:
        # 转发流式回复内容，空消息由发送方法忽略。
        async for chain in generator:
            await self.send(chain)


@register(PLUGIN_NAME, "ww", "小深信箱", "0.1.0")
class DeepLetterPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        counts = re.fullmatch(
            r"\s*([0-9]+)\s*/\s*([0-9]+)\s*",
            str(self.config.get("letter_count", "0/0")),
        )
        try:
            if counts is None:
                raise ValueError("累计次数格式无效")
            self._send_count, self._total_count = int(counts[1]), int(counts[2])
        except ValueError:
            self._send_count = self._total_count = 0
            logger.warning("信箱累计次数格式无效，使用 0/0；应为非负整数 x/y。")
        self.config["letter_count"] = f"{self._send_count}/{self._total_count}"
        # 最新一封用于校验回信方，完整往返记录用于保存；重载后均不保留。
        self._letters: dict[str, Letter] = {}
        self._letter_history: dict[str, list[Letter]] = {}
        # 收发信共用锁，保护编号、往返次数、冷却和累计计数。
        self._lock = asyncio.Lock()
        self._archive_locks: dict[Path, asyncio.Lock] = {}
        self._target_catalog: dict[str, dict[str, str]] = {}
        self._last_send_time: float | None = None
        self._last_id = 0
        self._runtime = LetterRuntime(self)

    async def initialize(self) -> None:
        # 发现可用会话，并启用 Runner 运行适配。
        try:
            for conversation in await self.context.get_db().get_conversations():
                umo = str(conversation.user_id or "")
                if (
                    not umo
                    or umo in self._target_catalog
                    or not self._config_allows_plugin(umo)
                ):
                    continue
                mapping = self.context.astrbot_config_mgr._load_conf_mapping(umo)
                self._target_catalog[umo] = {
                    "alias": str(conversation.title or umo),
                    "config": str(mapping.get("path", "default")),
                }
            from astrbot.core.provider.register import llm_tools

            tool = llm_tools.get_func("deep_letter")
            if tool:
                entries = [
                    f"{item['alias']} -> {umo} (配置: {item['config']})"
                    for umo, item in sorted(self._target_catalog.items())
                ]
                tool.description += "\n当前可用收信会话：" + (
                    "；".join(entries) or "暂无已记录的会话"
                )
        except Exception:
            logger.exception("信箱发现可用收信会话失败")
        self._runtime.install()

    async def terminate(self) -> None:
        # 卸载插件时撤销进程内的 Runner 适配。
        self._runtime.uninstall()

    @filter.on_llm_request()
    async def describe_runtime_mode(self, event, req) -> None:
        # 向当前模型请求补充实际运行能力，不修改历史对话。
        tools = getattr(req, "func_tool", None)
        if tools is None or tools.get_tool("deep_letter") is None:
            return
        self._runtime.check()
        if self._runtime.active:
            mode = "隐藏模式：调用和结果提示在群聊中隐藏；满足单工具、成功、无追加消息等条件时可直接结束，不会有下一次对话。"
        else:
            mode = (
                "原生模式：信箱工具调用和结果会在群聊中显示。"
                "推荐只使用 send 发信。收到信件后直接在当前群聊用正文简要说明。"
            )
        req.system_prompt = (
            (req.system_prompt or "") + "\n[小深信箱当前运行状态] " + mode
        )

    def _config_allows_plugin(self, umo: str) -> bool:
        try:
            plugins = self.context.astrbot_config_mgr.get_conf(umo).get(
                "plugin_set", ["*"]
            )
            return (
                not isinstance(plugins, list)
                or "*" in plugins
                or PLUGIN_NAME in plugins
            )
        except Exception:
            logger.exception("信箱读取目标会话配置失败")
            return False

    def _target(self, recipient: str) -> str:
        recipient = str(recipient or "").strip()
        if recipient not in self._target_catalog:
            matches = [
                umo
                for umo, item in self._target_catalog.items()
                if item["alias"] == recipient
            ]
            if len(matches) > 1:
                raise ValueError("收信方别名不唯一，请使用完整 UMO。")
            if matches:
                recipient = matches[0]
        if not recipient:
            raise ValueError("需要提供收信方 recipient。")
        MessageSession.from_str(recipient)
        if not self._config_allows_plugin(recipient):
            raise ValueError("目标会话未启用信件插件。")
        return recipient

    def _session_name(self, umo: str) -> str:
        return self._target_catalog.get(umo, {}).get("alias", umo)

    def _expire_letters(self) -> None:
        # 发新信或广播时按首次编号清理过期内存记录，回信不续期，磁盘存档不受影响。
        cutoff = time.time_ns() // 1_000_000 - LETTER_TTL_SECONDS * 1000
        for key in list(self._letters):
            if int(key) <= cutoff:
                self._letters.pop(key, None)
                self._letter_history.pop(key, None)

    async def _enqueue(self, letter: Letter) -> None:
        session = MessageSession.from_str(letter.target)
        platform = self.context.get_platform_inst(session.platform_id)
        if platform is None:
            raise ValueError("目标平台未运行。")
        source_name = self._session_name(letter.source)
        target_name = self._session_name(letter.target)
        text = (
            f"[来自其他群聊的信件：{source_name} → {target_name}]\n"
            f"编号：{letter.letter_id} 次数：{letter.round_no}/{MAX_ROUNDS}\n"
            + (
                "[广播信：无需回信]隐藏模式调用 silent 结束，原生模式用正文简单输出"
                if letter.is_broadcast
                else ""
            )
            + f"{letter.content}"
        )
        event = InternalLetterEvent(
            self.context, session, platform.meta(), text, source_name, target_name
        )
        # 入队成功不代表对方已处理；此处不经 QQ 接口发送信件。
        self.context.get_event_queue().put_nowait(event)

    async def _communicate(
        self,
        source: str,
        action: str,
        recipient: str,
        letter_id: str,
        content: str,
        *,
        _skip_cooldown: bool = False,
    ) -> dict:
        # 在调用方持有通信锁时校验信件并加入事件队列。
        # _skip_cooldown 仅供同一锁内已检查冷却的广播使用。
        text = str(content or "").strip()
        if not text:
            raise ValueError("信件正文不能为空。")
        if action == "send":
            if not _skip_cooldown:
                self._expire_letters()
            target = self._target(recipient)
            if target == source:
                raise ValueError("不能向当前会话自己发信，请选择其他收信方。")
            now = time.monotonic()
            if not _skip_cooldown and self._last_send_time is not None:
                remaining = SEND_COOLDOWN_SECONDS - (now - self._last_send_time)
                if remaining > 0:
                    raise ValueError(
                        f"全局发信冷却中，请 {int(remaining) + 1} 秒后再发。"
                    )
            # 毫秒时间戳作为整段往返的固定编号，同一毫秒内也保证不重复。
            self._last_id = max(time.time_ns() // 1_000_000, self._last_id + 1)
            letter = Letter(
                str(self._last_id), source, target, text, is_broadcast=_skip_cooldown
            )
        else:
            original = self._letters.get(str(letter_id).strip())
            if original is None:
                raise ValueError("找不到该信件；未存档的通信状态在重载后不保留。")
            if original.target != source:
                raise ValueError("当前会话不是该信件的收信方。")
            if original.round_no >= MAX_ROUNDS:
                raise ValueError("该信件已达到 10 封上限。")
            target = original.source
            if not self._config_allows_plugin(target):
                raise ValueError("目标会话未启用信件插件。")
            letter = Letter(
                original.letter_id,
                source,
                target,
                text,
                original.round_no + 1,
            )
        # 先登记最新状态，入队失败则回滚，避免留下未发送的信件。
        previous = self._letters.get(letter.letter_id)
        self._letters[letter.letter_id] = letter
        try:
            await self._enqueue(letter)
        except BaseException:
            if previous is None:
                self._letters.pop(letter.letter_id, None)
            else:
                self._letters[letter.letter_id] = previous
            raise
        self._letter_history.setdefault(letter.letter_id, []).append(letter)
        if action == "send":
            self._last_send_time = time.monotonic()
            self._send_count += 1
        self._total_count += 1
        self.config["letter_count"] = f"{self._send_count}/{self._total_count}"
        # 直接保存计数，不触发插件重载，避免丢失内存中的往返记录。
        try:
            save_config = getattr(self.config, "save_config", None)
            if callable(save_config):
                save_config()
        except Exception:
            # 信件已入队，计数保存失败不能当作发送失败，否则容易重复发信。
            logger.exception("信件已入队，但累计次数保存失败；当前内存计数保留。")
        return {
            "letter_id": letter.letter_id,
            "round": letter.round_no,
            "status": "queued",
            "message": "已入队，尚未确认对方处理。",
        }

    async def _broadcast(self, source: str, content: str) -> dict:
        # 在调用方持有通信锁时，向各目标发送独立编号的广播信。
        self._expire_letters()
        text = str(content or "").strip()
        if not text:
            raise ValueError("信件正文不能为空。")
        if self._last_send_time is not None:
            remaining = SEND_COOLDOWN_SECONDS - (
                time.monotonic() - self._last_send_time
            )
            if remaining > 0:
                raise ValueError(f"全局发信冷却中，请 {int(remaining) + 1} 秒后再发。")
        targets = set(self._target_catalog)
        targets.discard(source)
        targets.discard("")
        # 广播各目标独立编号；部分成功后不回滚，也不重发整批。
        queued, skipped, failed = [], [], []
        for target in sorted(targets):
            if not self._config_allows_plugin(target):
                skipped.append({"recipient": target, "reason": "目标未启用信箱"})
                continue
            try:
                result = await self._communicate(
                    source,
                    "send",
                    target,
                    "",
                    text,
                    _skip_cooldown=True,
                )
            except Exception as exc:
                failed.append({"recipient": target, "error": str(exc)})
            else:
                queued.append({"recipient": target, **result})
        return {
            "ok": bool(queued) and not failed,
            "status": "partial"
            if queued and failed
            else "queued"
            if queued
            else "failed",
            "queued": queued,
            "skipped": skipped,
            "failed": failed,
            "message": (
                "成功仅表示入队。每个目标使用独立编号，可各自保存；广播默认无需回信。"
                "不要重复广播；部分失败时仅按失败目标决定后续处理。"
                if queued
                else "没有信件成功入队，未消耗发信冷却和计数。"
            ),
        }

    @staticmethod
    def _archive_path(
        folder: Path, title: str, current: Path | None = None, *, date_prefix: str = ""
    ) -> Path:
        # 选择可用的存档文件名，避免覆盖其他存档。
        title = BAD_FILENAME_CHARS.sub("_", str(title or "").strip()).rstrip(" .")[:80]
        if not title:
            raise ValueError("需要提供有效的存档标题。")
        if current is not None:
            match = re.match(r"^([0-9]{4}-[0-9]{2}-[0-9]{2})_", current.name)
            if match is None:
                raise ValueError("存档文件名缺少日期前缀，无法改名。")
            prefix = match.group(1)
        else:
            if not date_prefix:
                raise ValueError("新建存档必须提供首次发信日期。")
            prefix = date_prefix
        index = 1
        while True:
            suffix = "" if index == 1 else f"_{index}"
            # Linux 按字节限制文件名长度，不能只按字符数量截断。
            budget = 255 - len(f"{prefix}_{suffix}.md".encode())
            shortened = title.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
            path = folder / f"{prefix}_{shortened.rstrip(' .')}{suffix}.md"
            if path == current or not (path.exists() or path.is_symlink()):
                return path
            index += 1

    async def _archive(
        self,
        source: str,
        action: str,
        letter_id: str,
        content: str,
        name: str,
        new_name: str,
        recipient: str,
        month: str = "",
    ) -> dict:
        # 在实际存档目录的锁内管理所选月份的文件。
        selected_month = str(month or "").strip() or datetime.now().strftime("%Y-%m")
        if not re.fullmatch(
            r"[0-9]{4}-(0[1-9]|1[0-2])", selected_month
        ) or selected_month.startswith("0000-"):
            raise ValueError("month 必须为有效年月 YYYY-MM，例如 2026-10。")
        root = await resolve_workspace_root_for_umo(source)
        folder = root / "letters" / selected_month
        folder = folder.resolve()
        # 同一实际目录串行操作，独立工作区或月份可以分别处理。
        lock = self._archive_locks.setdefault(folder, asyncio.Lock())
        async with lock:
            folder.mkdir(parents=True, exist_ok=True)
            if action == "list":
                result = {
                    "files": sorted(
                        p.name
                        for p in folder.glob("*.md")
                        if p.is_file() and not p.is_symlink()
                    )
                }
            elif action == "save":
                if content or new_name or recipient:
                    raise ValueError(
                        "save 接受 letter_id 和可选 name，不接受内容、新标题或收信方。"
                    )
                key = str(letter_id or "").strip()
                # 后续没有 await，先复制已入队记录，再同步保存完整往返内容。
                history = list(self._letter_history.get(key, ()))
                if not history or source not in {
                    history[0].source,
                    history[0].target,
                }:
                    raise ValueError(
                        "缺少当前会话可用的信件编号；重载后通信记录不保留。"
                    )
                text = (
                    f"# 信件 {key}\n\n"
                    + "\n\n".join(
                        f"## 第 {item.round_no} 封：{self._session_name(item.source)} → {self._session_name(item.target)}\n\n{item.content}"
                        for item in history
                    )
                    + "\n"
                )
                # 按文件首行的原始编号查重，改名后重复保存仍能找到原文件。
                existing = None
                for candidate in folder.glob("*.md"):
                    if candidate.is_symlink() or not candidate.is_file():
                        continue
                    try:
                        with candidate.open(encoding="utf-8") as saved:
                            if saved.readline().strip() == f"# 信件 {key}":
                                existing = candidate
                                break
                    except UnicodeDecodeError:
                        continue
                # 重复保存保留文件名；首次保存的日期取自原编号的时间戳。
                path = existing or self._archive_path(
                    folder,
                    name or f"信件_{key}",
                    date_prefix=datetime.fromtimestamp(int(key) / 1000).strftime(
                        "%Y-%m-%d"
                    ),
                )
                temporary = folder / f".{uuid.uuid4().hex}.tmp"
                try:
                    with temporary.open("x", encoding="utf-8") as output:
                        output.write(text)
                    if existing is not None:
                        temporary.chmod(stat.S_IMODE(existing.stat().st_mode))
                    # 临时文件写完再替换，避免中断写入破坏原存档。
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                result = {
                    "file": path.name,
                    "letter_id": key,
                    "letters": len(history),
                    "status": "saved",
                }
            else:
                filename = str(name or "").strip()
                if not filename or "/" in filename or "\\" in filename:
                    raise ValueError("请提供 list 返回的完整文件名。")
                path = folder / filename
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.suffix.lower() != ".md"
                ):
                    raise ValueError("找不到指定月份的信件文件。")
                if action == "read":
                    result = {
                        "file": path.name,
                        "content": path.read_text(encoding="utf-8"),
                    }
                elif action == "delete":
                    path.unlink()
                    result = {"file": filename}
                else:
                    new_path = self._archive_path(folder, new_name, current=path)
                    if new_path != path:
                        path.rename(new_path)
                    result = {"file": new_path.name}

            return result

    @llm_tool(name="deep_letter")
    async def deep_letter(
        self,
        event: AstrMessageEvent,
        action: str,
        recipient: str = "",
        letter_id: str = "",
        content: str = "",
        name: str = "",
        new_name: str = "",
        silent: bool = True,
        month: str = "",
    ) -> str:
        """小深信箱，当你感觉有值得分享或保存的信息时自主使用，用户在隐藏模式下看不见这些操作。

        send 发信，broadcast 广播。共享冷却60秒，不能发自己。
        reply 回信，同编号最多10次。需要主动静默结束回信使用silent，收到广播直接使用silent。
        信件内容建议不超过100字，成功仅表示入队，不保证对方已处理。

        save 按编号保存；name 指定首次标题。
        list 列出存档，read 读取，rename 改名，delete 删除。
        month 可选年月 YYYY-MM，默认当月。

        该工具的全部调用默认隐藏，在无后续时调用默认会结束当前对话。
        需要继续对话时传入silent=false；read/list 始终继续。
        正文部分会全部发出到聊天界面中，需要静默时不要输出正文只调用工具。

        Args:
            action(string): send、broadcast、reply、save、read、list、rename、delete 或 silent。
            recipient(string): send 必填，使用已列出的会话别名或完整 UMO；broadcast/reply 不填。
            letter_id(string): reply/save 必填，沿用首次发信生成的编号。
            content(string): send/broadcast/reply 的正文，建议不超过100字；save 不接受。
            name(string): save 的可选首次标题；read/rename/delete 使用 list 返回的完整文件名。
            new_name(string): rename 的新标题，不含日期或扩展名。
            month(string): 存档操作的可选年月 YYYY-MM，默认当月。
            silent(boolean): 默认 true；确需操作后继续才设 false。read/list/silent 动作忽略此值。

        Returns:
            JSON 结果，含 ok、action、runtime_mode、auto_end_available 和操作结果或 error。
        """
        action = str(action or "").strip().lower()
        try:
            source = str(event.unified_msg_origin or "").strip()
            if not source:
                raise ValueError("无法确定当前会话。")
            if action not in ACTIONS:
                raise ValueError(
                    "action 必须是 send/broadcast/reply/save/rename/delete/read/list/silent。"
                )
            if not isinstance(silent, bool):
                raise ValueError("silent 必须是布尔值。")
            if month and action not in {"save", "read", "list", "rename", "delete"}:
                raise ValueError("month 仅用于 save/read/list/rename/delete。")
            if action not in {"send", "broadcast", "reply"}:
                logger.info("[%s] 调用信箱工具：%s", self._session_name(source), action)
            if action == "silent":
                result = {"status": "silent"}
            elif action == "broadcast":
                if any(
                    str(value or "").strip()
                    for value in (recipient, letter_id, name, new_name)
                ):
                    raise ValueError(
                        "broadcast 只接受 content 和可选 silent，不填写 recipient、letter_id 或标题。"
                    )
                async with self._lock:
                    result = await self._broadcast(source, content)
            elif action in {"send", "reply"}:
                async with self._lock:
                    result = await self._communicate(
                        source, action, recipient, letter_id, content
                    )
            else:
                result = await self._archive(
                    source, action, letter_id, content, name, new_name, recipient, month
                )
            self._runtime.check()
            return json.dumps(
                {
                    "ok": True,
                    "action": action,
                    **result,
                    "runtime_mode": "hidden" if self._runtime.active else "native",
                    "auto_end_available": self._runtime.active,
                },
                ensure_ascii=False,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "action": action,
                    "error": str(exc),
                    "runtime_mode": "hidden" if self._runtime.active else "native",
                    "auto_end_available": self._runtime.active,
                },
                ensure_ascii=False,
            )
