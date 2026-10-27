from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import MessageChain, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import At, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, Group, MessageMember
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType
from astrbot.core.workspace import resolve_workspace_root_for_umo

PLUGIN_NAME = "astrbot_plugin_deep_letter"
MAX_LETTERS = 10
COOLDOWN_MS = 60_000
TTL_MS = 86_400_000
BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
SHORT_PREFIX = re.compile(r"^(?:QQ|微信)?(?:群聊|私聊|会话)_")
SHORT_NAME = re.compile(r"^(?:QQ|微信)?(?:群聊|私聊|会话)_[0-9a-f]{4,64}$")


@dataclass
class Contact:
    umo: str
    short: str
    name: str = ""
    trusted: bool = False
    discovered: bool = False

    @property
    def label(self) -> str:
        return f"{self.name}（{self.short}）" if self.name else self.short


@dataclass
class Letter:
    source: str
    target: str
    content: str
    sender_id: str
    source_label: str
    target_label: str


@dataclass
class Broadcast:
    letters: list[Letter] = field(default_factory=list)


@dataclass
class Thread:
    history: list[Letter] = field(default_factory=list)


# 把信件包装成消息，交给对方的正常管道处理。
class LetterEvent(AstrMessageEvent):
    def __init__(self, context, session, platform_meta, text, sender_id, target_name):
        message = AstrBotMessage()
        message.type = session.message_type
        message.self_id = "deep_letter"
        message.session_id = session.session_id
        message.message_id = uuid.uuid4().hex
        message.sender = MessageMember(user_id=sender_id)
        if session.message_type == MessageType.GROUP_MESSAGE:
            message.group = Group(group_id=session.session_id, group_name=target_name)
            # 群聊需要 At 唤醒；私聊不需要，避免部分平台渲染出空占位。
            message.message = [At(qq=message.self_id), Plain(text)]
        else:
            message.message = [Plain(text)]
        message.message_str = text
        message.raw_message = text
        super().__init__(text, message, platform_meta, session.session_id)
        self._context = context
        self.is_wake = self.is_at_or_wake_command = True

    async def send(self, message: MessageChain) -> None:
        if message is None or not message.chain:
            return
        if not await self._context.send_message(self.session, message):
            raise RuntimeError("目标平台未运行。")
        await super().send(message)

    async def send_streaming(self, generator, use_fallback=False) -> None:
        # 仅兜底宿主接口；正常信件经工具入队后由 send 单块发出，不走此处。
        async for chain in generator:
            await self.send(chain)


@register(PLUGIN_NAME, "ww", "小深信箱", "0.1.1")
class DeepLetterPlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self.contacts: dict[str, Contact] = {}
        self.aliases: dict[str, str] = {}
        self.threads: dict[str, Thread] = {}
        self.broadcasts: dict[str, Broadcast] = {}
        self._lock = asyncio.Lock()
        self._archive_lock = asyncio.Lock()
        self._last_id = 0
        self._last_send: dict[str, int] = {}
        self._dirty = False
        self._save_task: asyncio.Task | None = None

    # 会话配置与联系人。
    def _enabled(self, umo: str) -> bool:
        plugins = self.context.get_config(umo).get("plugin_set", ["*"])
        return plugins is None or "*" in plugins or PLUGIN_NAME in plugins

    def _show(self, umo: str) -> bool:
        return bool(
            self.context.get_config(umo)
            .get("provider_settings", {})
            .get("show_tool_use_status", False)
        )

    def _prefix(self, umo: str) -> str:
        session = MessageSession.from_str(umo)
        platform = self.context.get_platform_inst(session.platform_id)
        kind = platform.meta().name.lower() if platform else ""
        if not kind:
            # 插件初始化可能早于平台实例，回退到默认平台配置。
            kind = next(
                (
                    str(item.get("type", "")).lower()
                    for item in self.context.get_config(None).get("platform", [])
                    if item.get("id") == session.platform_id
                ),
                "",
            )
        brand = (
            "QQ" if kind in {"aiocqhttp", "qq_official", "qq_official_webhook"} else ""
        )
        if "wechat" in kind or "weixin" in kind:
            brand = "微信"
        category = {
            MessageType.GROUP_MESSAGE: "群聊",
            MessageType.FRIEND_MESSAGE: "私聊",
        }.get(session.message_type, "会话")
        return brand + category

    async def initialize(self) -> None:
        configured = self.config.get("contacts", {})
        if not isinstance(configured, dict):
            raise ValueError("contacts 必须是别名到 UMO 的字典。")
        # 先读取配置联系人，同一会话可以有多个别名。
        trusted: dict[str, list[str]] = {}
        for name, umo in configured.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("联系人名称不能为空。")
            name = name.strip()
            MessageSession.from_str(umo)
            # 短编号前缀保留，格式不对时不能作为普通别名绕过校验。
            if SHORT_PREFIX.match(name) and not SHORT_NAME.fullmatch(name):
                raise ValueError("联系人短编号须为会话前缀加4至64位小写十六进制字符。")
            if SHORT_NAME.fullmatch(name) and not hashlib.sha256(
                umo.encode()
            ).hexdigest().startswith(name.rsplit("_", 1)[1]):
                raise ValueError("联系人短编号与目标会话不匹配。")
            trusted.setdefault(umo, []).append(name)
        # 只扫描已开启信箱的会话，优先取非空名称。
        scanned: dict[str, str] = {}
        for item in await self.context.get_db().get_conversations():
            umo = str(item.user_id or "")
            try:
                MessageSession.from_str(umo)
                if self._enabled(umo):
                    title = str(item.title or "").strip()
                    if not scanned.get(umo):
                        scanned[umo] = title
            except (ValueError, TypeError):
                continue
        # 配置名称优先，扫描到的重名不作为别名。
        names = Counter(
            name for umo, name in scanned.items() if umo not in trusted and name
        )
        claimed = {name for values in trusted.values() for name in values}
        if sum(len(values) for values in trusted.values()) != len(claimed):
            raise ValueError("配置联系人名称去除首尾空格后不能重复。")
        digests = {
            umo: hashlib.sha256(umo.encode()).hexdigest()
            for umo in trusted.keys() | scanned.keys()
        }
        # 短编号从四位开始，遇到冲突就延长。
        for umo, digest in sorted(digests.items()):
            size = 4
            while any(
                other != umo and value[:size] == digest[:size]
                for other, value in digests.items()
            ):
                size += 1
            candidate = scanned.get(umo, "")
            if (
                candidate in claimed
                or names[candidate] != 1
                or SHORT_NAME.fullmatch(candidate)
                or umo in candidate
            ):
                candidate = ""
            name = trusted[umo][0] if umo in trusted else candidate
            if SHORT_NAME.fullmatch(name):
                name = ""
            contact = Contact(
                umo,
                f"{self._prefix(umo)}_{digest[:size]}",
                name,
                umo in trusted,
                umo in scanned,
            )
            self.contacts[umo] = contact
            for alias in [
                contact.short,
                *trusted.get(umo, []),
                *([candidate] if candidate and umo not in trusted else []),
            ]:
                self.aliases[alias] = umo
        # 完整展示标签也可作为收信别名。
        for contact in self.contacts.values():
            self.aliases.setdefault(contact.label, contact.umo)
        # 空配置首次填入扫描结果；重名或无名称的会话使用短编号。
        if not configured and self.contacts:
            self.config["contacts"] = {
                contact.name or contact.short: contact.umo
                for contact in self.contacts.values()
            }
            for contact in self.contacts.values():
                contact.trusted = True
            self._dirty = True
            await self._flush_config()

    async def terminate(self) -> None:
        if self._save_task:
            await self._save_task

    # 为每次请求单独生成工具描述。
    @filter.on_llm_request()
    async def append_tool_description(self, event, req) -> None:
        tools = getattr(req, "func_tool", None)
        tool = tools.get_tool("deep_letter") if tools else None
        if tool is None:
            return
        show = self._show(event.unified_msg_origin)
        lines = [
            "[小深信箱] 当前会话工具调用显示开启，忽略基础描述中的自主使用建议。仅在用户明确要求时发信或广播；收到来信后不自主回信，按需向当前会话汇报。"
            if show
            else "[小深信箱] 当前会话调用工具显示未开启，可以主动向其他会话中的自己发信。成功默认返回 None 结束本次处理，需要继续时设 continue_chat=true。",
            "会话：",
        ]
        # 配置联系人置顶，按情况追加状态标记。
        for contact in sorted(self.contacts.values(), key=lambda c: not c.trusted):
            flags = []
            if contact.trusted:
                flags.append("已配置")
            if self._show(contact.umo):
                flags.append("显示")
            if contact.trusted and not contact.discovered:
                flags.append("未发现")
            lines.append(
                "- " + contact.label + ("：" + " ".join(flags) if flags else "")
            )
        # 仅复制本次请求的工具集合和目标工具，避免会话之间串描述。
        local_tools = copy.copy(tools)
        local_tools.tools = list(tools.tools)
        local_tool = copy.copy(tool)
        base = tool.description.split("\n[小深信箱]", 1)[0]
        local_tool.description = base + "\n" + "\n".join(lines)
        local_tools.add_tool(local_tool)
        req.func_tool = local_tools

    # 后台保存计数，不阻塞发信。
    async def _flush_config(self) -> None:
        while self._dirty:
            self._dirty = False
            try:
                save_async = getattr(self.config, "save_config_async", None)
                if save_async:
                    await save_async()
                else:
                    save = getattr(self.config, "save_config", None)
                    if save:
                        await asyncio.to_thread(save)
            except Exception:
                logger.error("信箱配置保存失败，当前内存配置保留；不会重新发送。")

    def _count(self, new: bool) -> None:
        try:
            left, right = str(self.config.get("counters", "0/0")).split("/")
            left, right = int(left), int(right)
        except (ValueError, TypeError):
            left, right = 0, 0
        self.config["counters"] = f"{left + int(new)}/{right + 1}"
        self._dirty = True
        if self._save_task is None or self._save_task.done():
            self._save_task = asyncio.create_task(self._flush_config())

    # 内存往返、冷却和消息入队。
    def _label(self, umo: str) -> str:
        contact = self.contacts.get(umo)
        return contact.label if contact else "当前会话"

    def _clean(self) -> None:
        cutoff = time.time_ns() // 1_000_000 - TTL_MS
        for records in (self.threads, self.broadcasts):
            for key in list(records):
                if int(key) <= cutoff:
                    del records[key]

    def _cooldown(self, source: str) -> None:
        remaining = COOLDOWN_MS - (
            time.time_ns() // 1_000_000 - self._last_send.get(source, 0)
        )
        if remaining > 0:
            raise ValueError(f"发信冷却中，请 {(remaining + 999) // 1000} 秒后再发。")

    def _new_id(self) -> str:
        self._last_id = max(time.time_ns() // 1_000_000, self._last_id + 1)
        return str(self._last_id)

    def _enqueue(self, letter: Letter, key: str, number: int, broadcast=False) -> None:
        if not self._enabled(letter.target):
            raise ValueError("目标会话未启用信箱。")
        session = MessageSession.from_str(letter.target)
        platform = self.context.get_platform_inst(session.platform_id)
        if platform is None:
            raise ValueError("目标平台未运行。")
        marker = "1/1 广播" if broadcast else f"{number}/{MAX_LETTERS}"
        text = (
            f"[来自其他会话的信件：{letter.source_label} → {letter.target_label}]"
            + f"\n编号：{key} 次数：{marker}\n{letter.content}"
        )
        incoming = LetterEvent(
            self.context,
            session,
            platform.meta(),
            text,
            letter.sender_id,
            letter.target_label,
        )
        try:
            self.context.get_event_queue().put_nowait(incoming)
        except Exception:
            raise ValueError("信件入队失败。") from None

    def _letter(self, event, target: str, content: str) -> Letter:
        source = event.unified_msg_origin
        return Letter(
            source,
            target,
            content,
            str(event.get_sender_id() or ""),
            self._label(source),
            self._label(target),
        )

    def _deliver(self, event, target: str, content: str, key: str = "") -> dict:
        if event.unified_msg_origin == target:
            raise ValueError("不能向当前会话自己发信。")
        key = key or self._new_id()
        thread = self.threads.get(key) or Thread()
        letter = self._letter(event, target, content)
        number = len(thread.history) + 1
        # 入队和登记之间没有 await，失败时不留下记录。
        self._enqueue(letter, key, number)
        thread.history.append(letter)
        self.threads[key] = thread
        return {"id": key, "n": number}

    async def _communicate(
        self,
        event: AstrMessageEvent,
        action: str,
        recipient: str,
        key: str,
        content: str,
    ) -> dict:
        async with self._lock:
            source = event.unified_msg_origin
            # 只有新信和广播检查冷却、清理过期记录。
            if action in {"send", "broadcast"}:
                self._clean()
                self._cooldown(source)
            if action == "send":
                target = self.aliases.get(recipient)
                if target is None:
                    raise ValueError(
                        "找不到收信别名，请使用联系人列表中的名称或短编号。"
                    )
                result = self._deliver(event, target, content)
                self._last_send[source] = int(result["id"])
                self._count(True)
                return result
            if action == "reply":
                if key in self.broadcasts:
                    raise ValueError("广播不支持回信，请另行发信。")
                thread = self.threads.get(key)
                if thread is None:
                    raise ValueError("找不到该编号的信件。")
                # 只有最新收信方可以回信。
                last = thread.history[-1]
                if last.target != source:
                    raise ValueError("当前会话不是最新一封信的收信方。")
                if len(thread.history) >= MAX_LETTERS:
                    raise ValueError("这封信已经到达上限，不能再回了。")
                result = self._deliver(event, last.source, content, key)
                self._count(False)
                return result
            targets = [target for target in self.contacts if target != source]
            if not targets:
                raise ValueError("没有可广播的其他会话。")
            key = self._new_id()
            # 广播按动作消耗一次冷却；全部失败也不自动重试。
            self._last_send[source] = int(key)
            record = Broadcast()
            failed = []
            for target in targets:
                letter = self._letter(event, target, content)
                try:
                    self._enqueue(letter, key, 1, broadcast=True)
                    record.letters.append(letter)
                except ValueError as exc:
                    failed.append(
                        {"to": self.contacts[target].short, "error": str(exc)}
                    )
            if record.letters:
                self.broadcasts[key] = record
                self._count(True)
            return {
                "id": key,
                "queued": len(record.letters),
                "failed": failed,
            }

    # 保存到参与会话自己的工作区。
    async def _save(self, source: str, key: str, name: str) -> dict:
        async with self._archive_lock:
            broadcast = self.broadcasts.get(key)
            if broadcast:
                # 发起方保存全部成功目标，收件方只保存自己的那份。
                letters = broadcast.letters
                history = (
                    list(letters)
                    if source == letters[0].source
                    else [letter for letter in letters if letter.target == source]
                )
            else:
                thread = self.threads.get(key)
                history = (
                    list(thread.history)
                    if thread
                    and source
                    in {
                        thread.history[0].source,
                        thread.history[0].target,
                    }
                    else []
                )
            if not history:
                raise ValueError("找不到当前会话可存档的信件。")
            root = await resolve_workspace_root_for_umo(source)
            if not root.is_dir():
                raise ValueError("当前会话没有工作区，无法存档。")
            return await asyncio.to_thread(
                self._write_archive, root, key, history, name, bool(broadcast)
            )

    @staticmethod
    def _write_archive(
        root: Path, key: str, history: list[Letter], name: str, broadcast: bool = False
    ) -> dict:
        date = datetime.fromtimestamp(int(key) / 1000)
        folder = root / "letters" / date.strftime("%Y-%m")
        folder.mkdir(parents=True, exist_ok=True)
        # 用文件首行编号查找旧存档，重复保存时刷新。
        existing = None
        for path in folder.glob("*.txt"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                with path.open(encoding="utf-8") as saved:
                    if saved.readline().strip() == key:
                        existing = path
                        break
            except UnicodeError:
                continue
        title = BAD_NAME.sub("_", name.strip()).rstrip(" .")
        title = title.encode("utf-8")[:160].decode("utf-8", errors="ignore")
        suffix = f"_{title}" if title else ""
        path = (
            existing
            if existing and not name
            else folder / (date.strftime("%d-%H-%M-%S") + suffix + ".txt")
        )
        if path != existing and (path.exists() or path.is_symlink()):
            path = folder / (
                date.strftime("%d-%H-%M-%S")
                + f"-{int(key) % 1000:03d}"
                + suffix
                + ".txt"
            )
            if path != existing and (path.exists() or path.is_symlink()):
                raise ValueError("存档名称冲突，请换一个。")
        if broadcast:
            first = history[0]
            body = f"{first.source_label}→{'、'.join(item.target_label for item in history)}\n{first.content}"
        else:
            body = "\n\n".join(
                f"{item.source_label}→{item.target_label}\n{item.content}"
                for item in history
            )
        text = f"{key}\n{history[0].sender_id}\n\n{body}\n"
        # 先写好临时内容，再改名并原子刷新，失败时保留旧内容。
        temporary = folder / f".{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(text, encoding="utf-8")
            if existing and existing != path:
                existing.rename(path)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return {"file": str(path.relative_to(root))}

    # 统一生成模型可见的精简结果。
    @staticmethod
    def _result(
        action: str, result: dict, show: bool, continue_chat: bool
    ) -> str | None:
        if action == "broadcast":
            output = {"id": result["id"], "queued": result["queued"]}
            if result["failed"]:
                output["failed"] = result["failed"]
        elif action == "save":
            output = {"file": result["file"]}
        else:
            output = {"id": result["id"], "n": result["n"]}
        if show:
            output["hint"] = "显示已开启"
        failed = action == "broadcast" and (
            not result["queued"] or bool(result["failed"])
        )
        if failed or continue_chat or show:
            return json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        return None

    # 工具入口只负责参数校验和动作分派。
    @filter.llm_tool(name="deep_letter")
    async def deep_letter(
        self,
        event: AstrMessageEvent,
        action: str,
        recipient: str = "",
        letter_id: str = "",
        content: str = "",
        name: str = "",
        continue_chat: bool = False,
    ) -> str | None:
        """小深信箱，当你感觉有值得向其他会话中的自己分享、转达或保存的信息时自主使用。
        send 发信，冷却60秒。reply 回信，原信与回信总计最多10封；不必回信时直接 stop，或者按情况向当前会话汇报。
        broadcast 广播，仅在用户明确要求时使用，不支持回信。
        stop 静默结束当前回信，始终返回 None，不受显示开关或 continue_chat 影响。
        save 按编号存档，依赖当前会话的工作区
        信件内容建议不超过100字，发送成功只代表入队，不代表对方已处理。
        调用工具前不要重复输出信件正文。尽量减少自主使用发信的频率。

        Args:
            action(string): send、broadcast、reply、save 或 stop；stop 无需其他参数。
            recipient(string): send 必填，见联系人列表中的名称或短编号。
            letter_id(string): reply/save 必填，整段往返编号保持不变。
            content(string): send/broadcast/reply 必填，建议不超过100字。
            name(string): save 可选，提供时更新存档名称。
            continue_chat(boolean): 成功后是否继续对话，默认 false，失败或显示开启时自动返回。
        """
        # stop 直接结束，不读取配置或修改任何状态。
        if action == "stop":
            return None
        try:
            if action not in {"send", "broadcast", "reply", "save"}:
                raise ValueError("action 仅支持 send/broadcast/reply/save/stop。")
            if not isinstance(continue_chat, bool):
                raise ValueError("continue_chat 必须是布尔值。")
            if not all(
                isinstance(value, str)
                for value in (recipient, letter_id, content, name)
            ):
                raise ValueError("联系人、编号、正文与名称必须是字符串。")
            source = event.unified_msg_origin
            if not source:
                raise ValueError("无法确定当前会话。")
            recipient, letter_id, content = (
                recipient.strip(),
                letter_id.strip(),
                content.strip(),
            )
            if action in {"reply", "save"} and not letter_id:
                raise ValueError("需要提供信件编号。")
            # 操作前读取开关，避免入队成功后因读取失败而误报。
            show = self._show(source)
            if action == "save":
                if content or recipient:
                    raise ValueError("save 只接受编号、可选名称和 continue_chat。")
                result = await self._save(source, letter_id, name)
            else:
                if not content:
                    raise ValueError("信件正文不能为空。")
                if (
                    name
                    or (action != "send" and recipient)
                    or (action != "reply" and letter_id)
                ):
                    raise ValueError("当前动作包含不适用的参数。")
                result = await self._communicate(
                    event, action, recipient, letter_id, content
                )
            return self._result(action, result, show, continue_chat)
        except ValueError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        except Exception:
            logger.error("信箱操作失败，请检查目标平台、工作区或配置。")
            return json.dumps(
                {"error": "操作失败，请检查目标平台、工作区或配置。"},
                ensure_ascii=False,
            )
