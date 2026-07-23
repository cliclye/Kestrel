import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { apiUrl } from "./api";

type CatalogModel = {
  id: string;
  name: string;
  description: string;
  size_gb: number;
  license?: string;
  tags?: string[];
  source?: string;
  family?: string;
  status?: string;
  ram_gb?: number;
  hf_repo?: string;
  engine?: string;
  chat?: string;
  tier?: string;
};

type Installed = {
  id: string;
  path?: string | null;
  name?: string;
  ready?: boolean;
  engine?: string;
  family?: string;
  incomplete?: boolean;
  chat_ok?: boolean;
  chat_mode?: string;
  impostor?: boolean;
  size_bytes?: number;
  weight_bytes?: number;
  has_weights?: boolean;
  source?: string;
  backend?: string;
  description?: string;
};

type Msg = {
  role: "user" | "assistant";
  content: string;
  stats?: ChatStats;
};

type WindhoverStats = {
  decode_tok_s?: number;
  prefill_tok_s?: number;
  footprint_gb?: number;
  sparsity_pct?: number;
  bytes_per_tok?: number;
  au_hit_pct?: number;
  forwards?: number;
};

type ChatStats = {
  rss_mb?: number;
  latency_ms?: number;
  tokens_per_sec?: number;
  completion_tokens?: number;
  backend?: string;
  selected_model?: string;
  preview_model?: string;
  family?: string;
  windhover?: WindhoverStats;
  engine_active?: boolean;
  engine_error?: string;
  fallback_from?: string;
};

type PullProgress = {
  id: string;
  pct: number;
  message: string;
  bytes?: number;
};

type Tab = "library" | "chat" | "agent" | "advanced";

type AgentStep = {
  step: number;
  assistant?: string;
  tool_calls?: Array<Record<string, unknown>>;
  tool_results?: Array<Record<string, unknown>>;
  stats?: ChatStats;
  done?: boolean;
};

type AgentToolResult = {
  tool?: string;
  ok?: boolean;
  error?: string;
  path?: string;
  summary?: string;
  entries?: unknown[];
};

/** Strip tool-call soup from assistant text so the thread shows prose only. */
function agentVisibleReply(text: string, hasTools: boolean): string {
  let t = (text || "").trim();
  if (!t) return "";
  // Drop TOOL … END blocks
  t = t.replace(/(?:^|\n)TOOL\s+\w+[\s\S]*?(?:\nEND\b)/gi, "\n");
  // Drop fenced tool/json blocks
  t = t.replace(/```(?:tool|json)\s*\n[\s\S]*?```/gi, "\n");
  // Drop <tool>…</tool>
  t = t.replace(/<tool>[\s\S]*?<\/tool>/gi, "\n");
  t = t.replace(/\n{3,}/g, "\n\n").trim();
  if (hasTools && !t) return "";
  // Hide obvious regurgitated demo blurb when tools ran
  if (hasTools && /simple calculator|basic arithmetic operations/i.test(t) && t.length < 800) {
    return "";
  }
  return t;
}

/** Cursor-style one-liners for tool activity (collapsed by default). */
function toolActivityLabel(tr: AgentToolResult): string {
  const tool = String(tr.tool || "tool");
  const path = tr.path ? String(tr.path) : "";
  if (tr.ok === false) return `${tool} failed${path ? ` · ${path}` : ""}`;
  if (tool.includes("read")) return path ? `Read ${path}` : "Read file";
  if (tool.includes("write") || tool.includes("edit") || tool.includes("create")) {
    return path ? `Edited ${path}` : "Edited file";
  }
  if (tool.includes("list") || tool.includes("tree") || tool.includes("glob")) {
    const n = Array.isArray(tr.entries) ? tr.entries.length : 0;
    if (n > 0) return `Explored ${n} item${n === 1 ? "" : "s"}${path ? ` in ${path}` : ""}`;
    return path ? `Listed ${path}` : "Listed folder";
  }
  if (tr.summary) return String(tr.summary);
  return path ? `${tool} · ${path}` : tool;
}

function toolActivityGroup(results: AgentToolResult[]): string {
  if (!results.length) return "";
  const labels = results.map(toolActivityLabel);
  if (labels.length === 1) return labels[0];
  const reads = labels.filter((l) => l.startsWith("Read ")).length;
  const edits = labels.filter((l) => l.startsWith("Edited ")).length;
  const explores = labels.filter((l) => l.startsWith("Explored ") || l.startsWith("Listed ")).length;
  const parts: string[] = [];
  if (explores) parts.push(`Explored ${explores} path${explores === 1 ? "" : "s"}`);
  if (reads) parts.push(`Read ${reads} file${reads === 1 ? "" : "s"}`);
  if (edits) parts.push(`Edited ${edits} file${edits === 1 ? "" : "s"}`);
  const other = labels.length - explores - reads - edits;
  if (other > 0) parts.push(`${other} other`);
  return parts.join(", ");
}

const FAMILIES = [
  { id: "all", label: "All" },
  { id: "ollama", label: "Ollama" },
  { id: "mac", label: "Mac 16GB" },
  { id: "windhover", label: "Windhover" },
  { id: "gemma", label: "Gemma" },
  { id: "phi", label: "Phi" },
  { id: "qwen", label: "Qwen" },
  { id: "deepseek", label: "DeepSeek" },
  { id: "minimax", label: "MiniMax" },
  { id: "llama", label: "Llama" },
  { id: "mistral", label: "Mistral" },
  { id: "glm", label: "GLM" },
  { id: "kimi", label: "Kimi" },
] as const;

function matchInstalled(list: Installed[], id: string) {
  const key = id.replace("/", "__");
  return list.find(
    (m) => m.id === id || m.id === key || m.id?.endsWith(id.split("/").pop() || "")
  );
}

function isOllamaModel(m: { id?: string; source?: string; backend?: string; chat_mode?: string }) {
  return (
    m.source === "ollama" ||
    m.backend === "ollama" ||
    m.chat_mode === "ollama" ||
    String(m.id || "").startsWith("ollama/")
  );
}

function modelPickerLabel(m: Installed) {
  const name = m.name || m.id;
  return isOllamaModel(m) ? `Ollama · ${name.replace(/^ollama\//, "")}` : name;
}

function isMacSmall(m: CatalogModel) {
  return m.tier === "mac16" || m.source === "hf_small" || (m.tags || []).includes("mac16");
}

function statusBadge(m: CatalogModel) {
  if (isMacSmall(m)) return { cls: "mac", label: "Mac 16GB" };
  if (m.status === "ready" && m.chat === "preview") return { cls: "ready", label: "Preview" };
  if (m.status === "ready" && m.chat === "engine-oracle") return { cls: "demo", label: "Engine demo" };
  if (m.status === "download") return { cls: "download", label: "Download" };
  if (m.status === "ready") return { cls: "ready", label: "Ready" };
  return { cls: "download", label: m.status || "Download" };
}

function cleanChatText(text: string): string {
  let s = text;
  // Truncate runaway decode at the first turn-ending marker
  const markers = [
    "<|end|>",
    "<|im_end|>",
    "<|eot_id|>",
    "<end_of_turn>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|im_start|>",
  ];
  let cut: number | null = null;
  for (const m of markers) {
    const i = s.indexOf(m);
    if (i >= 0 && (cut === null || i < cut)) cut = i;
  }
  if (cut !== null) s = s.slice(0, cut);
  return s
    .replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, "")
    .replace(/<thinking\b[^>]*>[\s\S]*?<\/thinking>/gi, "")
    .replace(/<redacted_reasoning\b[^>]*>[\s\S]*?<\/redacted_reasoning>/gi, "")
    .replace(/<reason\b[^>]*>[\s\S]*?<\/reason>/gi, "")
    .replace(/<think\b[^>]*>[\s\S]*$/gi, "")
    .replace(/<thinking\b[^>]*>[\s\S]*$/gi, "")
    .replace(/<\/?(?:think|thinking|redacted_reasoning|reason)\s*>/gi, "")
    .replace(/^\[(?:wh|WH|CUDA|DSA|COLI|coli|windhover)\][^\n]*/gm, "")
    .replace(/^CATS sparsity[^\n]*/gim, "")
    .replace(/<\|[^|>]+?\|>/g, "")
    .replace(/<\/?s>/g, "")
    .replace(/<end_of_turn>/g, "")
    .replace(/<start_of_turn>\w*/g, "")
    .replace(/\[\/?INST\]/g, "")
    .replace(/<<SYS>>|<<\/SYS>>/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function MarkdownBody({ text }: { text: string }) {
  const cleaned = cleanChatText(text);
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        ),
        pre: ({ children }) => <pre className="md-pre">{children}</pre>,
        code: ({ className, children, ...props }) => {
          const inline = !className;
          return inline ? (
            <code className="md-inline-code" {...props}>
              {children}
            </code>
          ) : (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
      }}
    >
      {cleaned}
    </ReactMarkdown>
  );
}

function RailIcon({ name }: { name: Tab }) {
  const common = {
    width: 18,
    height: 18,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.75,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true as const,
  };
  if (name === "agent") {
    return (
      <svg {...common}>
        <path d="M12 3v3M12 18v3M3 12h3M18 12h3" />
        <circle cx="12" cy="12" r="4.5" />
      </svg>
    );
  }
  if (name === "chat") {
    return (
      <svg {...common}>
        <path d="M5 6.5h14a1.5 1.5 0 0 1 1.5 1.5v7A1.5 1.5 0 0 1 19 16.5H10L6 20v-3.5H5A1.5 1.5 0 0 1 3.5 15V8A1.5 1.5 0 0 1 5 6.5z" />
      </svg>
    );
  }
  if (name === "library") {
    return (
      <svg {...common}>
        <path d="M5 4.5h5.5v15H5zM13.5 4.5H19v15h-5.5z" />
      </svg>
    );
  }
  return (
    <svg {...common}>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 3.5v2.2M12 18.3v2.2M3.5 12h2.2M18.3 12h2.2M6.2 6.2l1.6 1.6M16.2 16.2l1.6 1.6M17.8 6.2l-1.6 1.6M7.8 16.2l-1.6 1.6" />
    </svg>
  );
}

export function App() {
  const [tab, setTab] = useState<Tab>("agent");
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [installed, setInstalled] = useState<Installed[]>([]);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [engineOk, setEngineOk] = useState<boolean | null>(null);
  const [enginePresent, setEnginePresent] = useState<boolean | null>(null);
  const [family, setFamily] = useState<string>("all");
  const [query, setQuery] = useState("");
  const [activeModel, setActiveModel] = useState<string>("");
  const [stats, setStats] = useState<{
    rss_mb?: number;
    engine?: string;
    engine_present?: boolean;
    installed?: number;
    last?: ChatStats;
    chat_preview?: string;
  } | null>(null);
  const [lastStats, setLastStats] = useState<ChatStats | null>(null);
  const [confirmUninstall, setConfirmUninstall] = useState<{ id: string; name: string; path?: string } | null>(
    null
  );
  const [progress, setProgress] = useState<PullProgress | null>(null);
  const [workspace, setWorkspace] = useState("");
  const [agentInput, setAgentInput] = useState("");
  const [agentBusy, setAgentBusy] = useState(false);
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([]);
  const [agentSummary, setAgentSummary] = useState("");
  const [agentPrompt, setAgentPrompt] = useState("");
  const [agentStatus, setAgentStatus] = useState("");
  const [agentPhase, setAgentPhase] = useState("");
  const [pickingFolder, setPickingFolder] = useState(false);
  const [tree, setTree] = useState<Array<{ name: string; path: string; type: string; size?: number | null }>>([]);
  const [workspaceReady, setWorkspaceReady] = useState(false);
  const [appVersion, setAppVersion] = useState("");
  const [updateInfo, setUpdateInfo] = useState<{
    available?: boolean;
    current?: string;
    latest?: string;
    download_url?: string | null;
    html_url?: string;
    error?: string | null;
    name?: string | null;
  } | null>(null);
  const [updateBusy, setUpdateBusy] = useState(false);
  const [updateMsg, setUpdateMsg] = useState("");
  const threadRef = useRef<HTMLDivElement>(null);
  const busyRef = useRef<string | null>(null);
  const progressRef = useRef<PullProgress | null>(null);

  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);
  useEffect(() => {
    progressRef.current = progress;
  }, [progress]);

  const chatCapable = useMemo(
    () => installed.filter((m) => m.chat_ok && !m.impostor),
    [installed]
  );

  const ollamaInstalled = useMemo(
    () => installed.filter((m) => isOllamaModel(m) && m.chat_ok),
    [installed]
  );

  async function refresh() {
    try {
      const health = await fetch(apiUrl("/health")).then((r) => r.json());
      setEngineOk(!!health?.ok);
      if (health?.version) setAppVersion(String(health.version));
      setEnginePresent(
        typeof health?.engine_present === "boolean"
          ? health.engine_present
          : typeof health?.engine_on === "boolean"
            ? health.engine_on
            : null
      );
      const [c, i, s] = await Promise.all([
        fetch(apiUrl("/v1/catalog")).then((r) => r.json()),
        fetch(apiUrl("/api/installed")).then((r) => r.json()),
        fetch(apiUrl("/api/stats"))
          .then((r) => r.json())
          .catch(() => null),
      ]);
      const models = c.models || [];
      const list = (i.data || []) as Installed[];
      setCatalog(models);
      setInstalled(list);
      if (s) {
        setStats(s);
        if (typeof s.engine_present === "boolean") setEnginePresent(s.engine_present);
        if (s.last && typeof s.last === "object") setLastStats(s.last as ChatStats);
      }
      if (health?.stats) setLastStats(health.stats);
      // Don't wipe download / uninstall status from the 5s poller
      if (!busyRef.current && !progressRef.current) setStatus("");
      setActiveModel((prev) => {
        const ok = list.filter((m) => m.chat_ok && !m.impostor);
        if (prev && ok.some((m) => m.id === prev || m.id === prev.replace("/", "__"))) return prev;
        return ok[0] ? String(ok[0].id) : "";
      });
    } catch {
      setEngineOk(false);
      setEnginePresent(false);
      try {
        const c = await fetch("./catalog.json").then((r) => r.json());
        setCatalog(c.models || []);
        setStatus("Engine offline — open the Mac app or run ./windhover app");
      } catch {
        setStatus("Start Windhover: open the Mac app, or run ./windhover app");
      }
    }
  }

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), 5000);
    return () => clearInterval(t);
  }, []);

  async function checkForUpdate(force = false) {
    try {
      const q = force ? "?force=1" : "";
      const j = await fetch(apiUrl(`/v1/update${q}`)).then((r) => r.json());
      setUpdateInfo(j);
      if (j?.current) setAppVersion(String(j.current));
      if (force && !j?.available) {
        setUpdateMsg(j?.error || (j?.latest ? `You're on the latest version (${j.latest}).` : "No update available."));
      }
      return j;
    } catch (e) {
      setUpdateMsg(`Update check failed: ${e instanceof Error ? e.message : String(e)}`);
      return null;
    }
  }

  async function applyUpdate() {
    if (updateBusy) return;
    setUpdateBusy(true);
    setUpdateMsg("Downloading update…");
    try {
      const j = await fetch(apiUrl("/v1/update/apply"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ download_url: updateInfo?.download_url || undefined }),
      }).then((r) => r.json());
      if (j?.ok) {
        setUpdateMsg(String(j.message || "Installer launched. Finish setup to upgrade."));
      } else {
        setUpdateMsg(String(j?.error || "Update failed."));
        if (updateInfo?.html_url) {
          window.open(updateInfo.html_url, "_blank", "noopener,noreferrer");
        }
      }
    } catch (e) {
      setUpdateMsg(`Update failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setUpdateBusy(false);
    }
  }

  useEffect(() => {
    const t = setTimeout(() => void checkForUpdate(false), 2500);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (family === "ollama") return [];
    return catalog.filter((m) => {
      if (family === "mac") {
        if (!isMacSmall(m)) return false;
      } else if (family !== "all" && (m.family || "other") !== family) {
        return false;
      }
      if (!q) return true;
      const hay = `${m.name} ${m.id} ${m.description} ${(m.tags || []).join(" ")}`.toLowerCase();
      return hay.includes(q);
    });
  }, [catalog, family, query]);

  const filteredOllama = useMemo(() => {
    if (family !== "all" && family !== "ollama") return [];
    const q = query.trim().toLowerCase();
    return ollamaInstalled.filter((m) => {
      if (!q) return true;
      const hay = `${m.name || ""} ${m.id} ${m.description || ""}`.toLowerCase();
      return hay.includes(q);
    });
  }, [ollamaInstalled, family, query]);

  const activeMeta = useMemo(
    () => matchInstalled(chatCapable, activeModel) || chatCapable[0],
    [chatCapable, activeModel]
  );

  async function pull(m: CatalogModel, weights = false) {
    const autoWeights = isMacSmall(m);
    let useWeights = weights || autoWeights;
    if (m.status === "download" && !useWeights) {
      const ok = confirm(
        `${m.name} is ~${m.size_gb} GB from Hugging Face.\n\n` +
          `Windhover will NOT install a fake stub. Continue with a real download?\n\n` +
          `For Mac 16GB local chat under 20GB, use the Mac 16GB filter instead.`
      );
      if (!ok) return;
      useWeights = true;
    }
    setBusy(m.id);
    setProgress({ id: m.id, pct: 1, message: `Starting ${m.name}…` });
    setStatus(
      useWeights || isMacSmall(m)
        ? `Downloading ${m.name} (~${m.size_gb} GB)…`
        : `Installing ${m.name}…`
    );
    try {
      const r = await fetch(apiUrl("/api/pull"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: m.id, weights: useWeights, async: true }),
      });
      const j = await r.json();
      if (j.async && j.job_id) {
        // Poll job progress until done/error
        for (;;) {
          await new Promise((res) => setTimeout(res, 450));
          const st = await fetch(apiUrl(`/api/jobs/${j.job_id}`)).then((x) => x.json());
          if (st.error === "unknown job") {
            setStatus("Lost download job — try again");
            setProgress(null);
            break;
          }
          const pct = Number(st.pct) || 0;
          const message = String(st.message || st.error || `Downloading ${m.name}…`);
          setProgress({
            id: m.id,
            pct,
            message,
            bytes: typeof st.bytes === "number" ? st.bytes : undefined,
          });
          setStatus(message);
          if (st.status === "done") {
            setStatus(`Installed ${m.name}`);
            setProgress({ id: m.id, pct: 100, message: "Installed" });
            if (isMacSmall(m) || m.chat === "engine") {
              setActiveModel(m.id);
            }
            break;
          }
          if (st.status === "error") {
            setStatus(st.error || `Failed: ${m.id}`);
            setProgress(null);
            break;
          }
        }
      } else if (j.ok) {
        setStatus(`Installed ${m.name}`);
        setProgress({ id: m.id, pct: 100, message: "Installed" });
        if (isMacSmall(m) || m.chat === "engine") {
          setActiveModel(m.id);
        }
      } else {
        setStatus(j.error || `Failed: ${m.id}`);
        setProgress(null);
      }
      await refresh();
      // Clear progress bar shortly after success
      setTimeout(() => setProgress((p) => (p?.id === m.id && p.pct >= 100 ? null : p)), 1200);
    } catch (e) {
      setStatus(String(e));
      setProgress(null);
    } finally {
      setBusy(null);
    }
  }

  async function uninstall(id: string, name?: string, path?: string) {
    // Prefer in-app confirm — window.confirm is unreliable in the Tauri webview.
    setConfirmUninstall({ id, name: name || id, path });
  }

  async function confirmUninstallNow() {
    if (!confirmUninstall) return;
    const { id, name, path } = confirmUninstall;
    setConfirmUninstall(null);
    setBusy(id);
    setStatus(`Uninstalling ${name}…`);
    try {
      const r = await fetch(apiUrl("/api/uninstall"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, path }),
      });
      const j = await r.json();
      if (j.ok) {
        setStatus(`Uninstalled ${name}`);
        if (activeModel === id || activeModel.replace("/", "__") === id.replace("/", "__")) {
          setActiveModel("");
          setMessages([]);
        }
      } else {
        setStatus(j.error || `Failed to uninstall ${id}`);
      }
      await refresh();
    } catch (e) {
      setStatus(String(e));
    } finally {
      setBusy(null);
    }
  }

  async function openChat(id: string) {
    const inst = matchInstalled(installed, id);
    if (!inst?.chat_ok) {
      setStatus("That pack can’t chat — install a Hugging Face model from Library (Mac 16GB), or start Ollama.");
      return;
    }
    setActiveModel(id);
    setMessages([]);
    setTab("chat");
  }

  async function send() {
    const text = input.trim();
    if (!text || sending) return;
    const modelId = activeModel || chatCapable[0]?.id;
    if (!modelId) {
      setStatus("Install a Hugging Face model from Library first (or start Ollama)");
      setTab("library");
      return;
    }
    setSending(true);
    setInput("");
    const next = [...messages, { role: "user" as const, content: text }];
    setMessages(next);
    try {
      const r = await fetch(apiUrl("/v1/chat/completions"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: modelId,
          messages: next.map(({ role, content }) => ({ role, content })),
          max_tokens: 512,
          temperature: 0.7,
        }),
      });
      const j = await r.json();
      const st = (j?.stats || {}) as ChatStats;
      if (typeof j?.engine_active === "boolean") st.engine_active = j.engine_active;
      if (j?.code === "engine_inactive" || j?.engine_active === false) {
        st.engine_active = false;
        if (j?.error) st.engine_error = String(j.error);
      }
      if (!r.ok || j?.code === "engine_inactive") {
        const errMsg =
          String(j?.error || `Chat failed (${r.status})`) +
          (j?.detail ? `\n\n${j.detail}` : "");
        setLastStats(st);
        setStatus(
          j?.code === "engine_inactive" || r.status === 503
            ? "Windhover engine is not active"
            : String(j?.error || "Chat failed")
        );
        setMessages([
          ...next,
          {
            role: "assistant",
            content: errMsg,
            stats: { ...st, engine_active: false, backend: st.backend || "error" },
          },
        ]);
        void refresh();
        return;
      }
      const content =
        j?.choices?.[0]?.message?.content ||
        j?.error ||
        "No response.";
      setLastStats(st);
      if (st.engine_active === false || st.fallback_from) {
        setStatus(st.engine_error || "Windhover engine inactive — reply used fallback");
      }
      setMessages([
        ...next,
        {
          role: "assistant",
          content: String(content),
          stats: st,
        },
      ]);
      void refresh();
    } catch (e) {
      setMessages([...next, { role: "assistant", content: String(e) }]);
      setStatus("Chat request failed — is Windhover running?");
    } finally {
      setSending(false);
    }
  }

  async function loadWorkspaceTree() {
    try {
      const treeR = await fetch(apiUrl("/api/workspace/tree?path=.")).then((x) => x.json());
      if (treeR.ok) {
        setTree(treeR.entries || []);
        setWorkspaceReady(true);
        return true;
      }
      setTree([]);
      setAgentStatus(treeR.error || "Could not list folder");
      return false;
    } catch (e) {
      setTree([]);
      setAgentStatus(String(e));
      return false;
    }
  }

  async function applyWorkspace(path?: string) {
    const root = (path ?? workspace).trim();
    if (!root) {
      // Empty field: open the native picker instead of failing silently.
      await browseWorkspace();
      return;
    }
    setAgentStatus(`Setting workspace…`);
    try {
      const r = await fetch(apiUrl("/api/workspace"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ root }),
      });
      const j = await r.json();
      if (!j.ok) {
        setWorkspaceReady(false);
        setTree([]);
        setAgentStatus(j.error || "Could not set workspace");
        setStatus(j.error || "Could not set workspace");
        return;
      }
      setWorkspace(String(j.root));
      setStatus(`Workspace: ${j.root}`);
      setAgentStatus(`Using ${j.root}`);
      await loadWorkspaceTree();
    } catch (e) {
      setWorkspaceReady(false);
      setAgentStatus(String(e));
      setStatus(String(e));
    }
  }

  async function browseWorkspace() {
    if (agentBusy || pickingFolder) return;
    setPickingFolder(true);
    setAgentStatus("Choose a folder…");
    try {
      const r = await fetch(apiUrl("/api/workspace/pick"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const j = await r.json();
      if (j.cancelled) {
        setAgentStatus("Folder picker cancelled");
        return;
      }
      if (!j.ok || !j.root) {
        setWorkspaceReady(false);
        setTree([]);
        setAgentStatus(j.error || "Could not pick folder");
        setStatus(j.error || "Could not pick folder");
        return;
      }
      setWorkspace(String(j.root));
      setStatus(`Workspace: ${j.root}`);
      setAgentStatus(`Using ${j.root}`);
      await loadWorkspaceTree();
    } catch (e) {
      setWorkspaceReady(false);
      setAgentStatus(String(e));
      setStatus(String(e));
    } finally {
      setPickingFolder(false);
    }
  }

  async function runAgent() {
    const text = agentInput.trim();
    if (!text || agentBusy) return;
    const modelId = activeModel || chatCapable[0]?.id;
    if (!modelId) {
      setStatus("Install a chat-capable model from Library first");
      setTab("library");
      return;
    }
    if (!workspace.trim()) {
      setStatus("Select a workspace folder first");
      return;
    }
    setAgentBusy(true);
    setAgentSteps([]);
    setAgentSummary("");
    setAgentPrompt(text);
    setAgentInput("");
    setAgentStatus("Starting…");
    setAgentPhase("start");
    setStatus("Agent running…");
    try {
      await applyWorkspace(workspace);
      const r = await fetch(apiUrl("/api/agent"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: modelId,
          workspace,
          prompt: text,
          max_steps: 8,
          max_tokens: 384,
          temperature: 0.15,
          async: true,
        }),
      });
      const j = await r.json();
      if (!j.ok) {
        setStatus(j.error || "Agent failed");
        setAgentSummary(String(j.error || "failed"));
        setAgentStatus(String(j.error || "failed"));
        return;
      }
      // Sync fallback (older servers)
      if (!j.async || !j.job_id) {
        setAgentSteps((j.steps || []) as AgentStep[]);
        setAgentSummary(String(j.summary || ""));
        setAgentStatus("Done");
        setAgentPhase("done");
        setStatus("Agent finished");
        return;
      }
      const jobId = String(j.job_id);
      for (;;) {
        await new Promise((res) => setTimeout(res, 400));
        const st = await fetch(apiUrl(`/api/jobs?id=${encodeURIComponent(jobId)}`)).then((x) =>
          x.json()
        );
        if (Array.isArray(st.steps)) setAgentSteps(st.steps as AgentStep[]);
        if (st.message) setAgentStatus(String(st.message));
        if (st.phase) setAgentPhase(String(st.phase));
        if (st.status === "done") {
          const result = st.result || {};
          setAgentSteps((result.steps || st.steps || []) as AgentStep[]);
          setAgentSummary(String(result.summary || st.summary || ""));
          setAgentStatus("Done");
          setAgentPhase("done");
          setStatus("Agent finished");
          break;
        }
        if (st.status === "error") {
          setAgentSummary(String(st.error || st.message || "Agent failed"));
          setAgentStatus(String(st.error || st.message || "failed"));
          setAgentPhase("error");
          setStatus(String(st.error || "Agent failed"));
          break;
        }
      }
      const treeR = await fetch(apiUrl("/api/workspace/tree?path=.")).then((x) => x.json());
      if (treeR.ok) setTree(treeR.entries || []);
    } catch (e) {
      setStatus(String(e));
      setAgentSummary(String(e));
      setAgentStatus(String(e));
      setAgentPhase("error");
    } finally {
      setAgentBusy(false);
    }
  }

  const engineTitle =
    engineOk === false
      ? "Windhover API unreachable"
      : enginePresent === false
        ? "API up, but windhover-engine binary missing — run ./windhover build"
        : lastStats?.engine_active === false
          ? lastStats.engine_error ||
            "Last reply did not use windhover-engine — engine inactive for that turn"
          : engineOk
            ? "Windhover API online · windhover-engine ready"
            : "Checking…";
  const engineState =
    engineOk === false
      ? "off"
      : enginePresent === false || lastStats?.engine_active === false
        ? "warn"
        : engineOk
          ? "on"
          : "";
  const engineInactiveBanner =
    engineOk === false
      ? "Windhover API offline — open the Mac app or run ./windhover app"
      : enginePresent === false
        ? "Windhover engine is not active — binary missing. Run ./windhover build, then restart."
        : lastStats?.engine_active === false
          ? lastStats.engine_error ||
            "Windhover engine is not active for the last reply (fallback or failure)."
          : null;

  const modelSelect = (
    <select
      value={activeModel}
      onChange={(e) => {
        setActiveModel(e.target.value);
        if (tab === "chat") setMessages([]);
      }}
      disabled={!chatCapable.length}
      aria-label="Model"
    >
      {!chatCapable.length ? (
        <option value="">Install a model or start Ollama</option>
      ) : null}
      {chatCapable.map((m) => (
        <option key={m.id} value={m.id}>
          {modelPickerLabel(m)}
        </option>
      ))}
    </select>
  );

  return (
    <div className={`shell${tab === "agent" ? " has-side" : ""}`}>
      {confirmUninstall && typeof document !== "undefined"
        ? createPortal(
            <div className="modal-backdrop" role="dialog" aria-modal="true">
              <div className="modal">
                <h2>Uninstall model?</h2>
                <p>
                  Remove <strong>{confirmUninstall.name}</strong> from this Mac? This deletes local
                  files under <code>~/.windhover/models</code>.
                </p>
                <div className="modal-actions">
                  <button type="button" className="btn ghost" onClick={() => setConfirmUninstall(null)}>
                    Cancel
                  </button>
                  <button
                    type="button"
                    className="btn primary danger-fill"
                    onClick={() => void confirmUninstallNow()}
                  >
                    Uninstall
                  </button>
                </div>
              </div>
            </div>,
            document.body
          )
        : null}

      <div className="shell-chrome">
        <header className="titlebar slim">
          <div className="traffic-spacer" aria-hidden />
          <span className="titlebar-label">Windhover</span>
          <div className="titlebar-end">
            {updateInfo?.available ? (
              <button
                type="button"
                className="btn update-btn"
                disabled={updateBusy}
                onClick={() => void applyUpdate()}
                title={`Update to ${updateInfo.latest}`}
              >
                {updateBusy ? "Updating…" : `Update to ${updateInfo.latest}`}
              </button>
            ) : null}
            <div className={`engine-pill compact ${engineState}`} title={engineTitle}>
              <i className="dot" aria-hidden />
              <strong>
                {engineOk === false
                  ? "Off"
                  : enginePresent === false
                    ? "No binary"
                    : engineOk
                      ? "Engine"
                      : "…"}
              </strong>
            </div>
          </div>
        </header>

        {updateInfo?.available ? (
          <div className="update-banner" role="status">
            <strong>Update available</strong>
            <span>
              Windhover {updateInfo.latest} is ready
              {updateInfo.current ? ` (you have ${updateInfo.current})` : ""}. Update in place — no
              uninstall needed.
            </span>
            <button
              type="button"
              className="btn primary"
              disabled={updateBusy}
              onClick={() => void applyUpdate()}
            >
              {updateBusy ? "Downloading…" : "Update now"}
            </button>
            {updateInfo.html_url ? (
              <a className="btn ghost" href={updateInfo.html_url} target="_blank" rel="noreferrer">
                Release notes
              </a>
            ) : null}
            {updateMsg ? <span className="update-msg">{updateMsg}</span> : null}
          </div>
        ) : null}
      </div>

      <div className="app-frame">
        <nav className="rail" aria-label="Primary">
          <button
            type="button"
            className="rail-brand"
            title="Windhover"
            onClick={() => setTab("agent")}
          >
            <img src="./windhover-icon.png" alt="" width={28} height={28} />
          </button>
          {(
            [
              { id: "agent" as Tab, label: "Agent" },
              { id: "chat" as Tab, label: "Chat" },
              { id: "library" as Tab, label: "Library" },
              { id: "advanced" as Tab, label: "Advanced" },
            ] as const
          ).map((item) => (
            <button
              key={item.id}
              type="button"
              className={`rail-btn${tab === item.id ? " active" : ""}`}
              aria-label={item.label}
              aria-current={tab === item.id ? "page" : undefined}
              title={item.label}
              onClick={() => setTab(item.id)}
            >
              <RailIcon name={item.id} />
            </button>
          ))}
          <div className="rail-spacer" />
          <div className={`rail-engine ${engineState}`} title={engineTitle} aria-hidden>
            <i className="dot" />
          </div>
        </nav>

        {tab === "agent" ? (
          <aside className="side-panel" aria-label="Workspace">
            <div className="side-head">
              <strong>Workspace</strong>
              <span className="muted">{workspaceReady ? "Ready" : "Pick a folder"}</span>
            </div>
            <div className="side-workspace">
              <label className="workspace-field">
                <span>Folder</span>
                <input
                  value={workspace}
                  onChange={(e) => {
                    setWorkspace(e.target.value);
                    setWorkspaceReady(false);
                  }}
                  placeholder="/path/to/project"
                  disabled={agentBusy || pickingFolder}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void applyWorkspace();
                    }
                  }}
                />
              </label>
              <div className="side-actions">
                <button
                  type="button"
                  className="btn ghost"
                  disabled={agentBusy || pickingFolder}
                  onClick={() => void browseWorkspace()}
                >
                  {pickingFolder ? "…" : "Browse"}
                </button>
                <button
                  type="button"
                  className="btn primary"
                  disabled={agentBusy || pickingFolder}
                  onClick={() => void applyWorkspace()}
                >
                  Use
                </button>
              </div>
            </div>
            {agentStatus ? <p className="side-status">{agentStatus}</p> : null}
            <div className="side-tree" aria-label="Workspace files">
              <strong>Files</strong>
              {!workspaceReady ? (
                <p className="muted">Browse or enter a path, then Use</p>
              ) : tree.length === 0 ? (
                <p className="muted">Folder is empty</p>
              ) : (
                <ul>
                  {tree.map((e) => (
                    <li key={e.path} className={e.type === "dir" ? "dir" : "file"}>
                      {e.type === "dir" ? "▸ " : ""}
                      {e.name}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </aside>
        ) : null}

        <main className={`work-main tab-${tab}`}>
        {tab === "library" ? (
          <div className="library-scroll">
            <section className="library-intro">
              <h1>Library</h1>
              <p className="lead">
                Install real Hugging Face models. Chat and Agent stay on your machine — no fake stand-ins.
              </p>
              <div className="hero-cta">
                <button
                  type="button"
                  className="btn primary"
                  onClick={() => {
                    const small = catalog.find((m) => isMacSmall(m));
                    if (small) void pull(small);
                    else setFamily("mac");
                  }}
                >
                  Install a Mac 16GB model
                </button>
                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => setTab("chat")}
                  disabled={!chatCapable.length}
                >
                  Open chat
                </button>
              </div>
            </section>

            <section className="library">
              <div className="toolbar">
                <div className="families" role="tablist" aria-label="Model families">
                  {FAMILIES.map((f) => (
                    <button
                      key={f.id}
                      type="button"
                      role="tab"
                      aria-selected={family === f.id}
                      className={family === f.id ? "active" : ""}
                      onClick={() => setFamily(f.id)}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
                <label className="search">
                  <span className="sr">Search</span>
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search Qwen, Kimi, GLM…"
                  />
                </label>
              </div>

              {status ? (
                <p className="status">{status}</p>
              ) : (
                <p className="status muted">
                  {family === "ollama"
                    ? `${filteredOllama.length} Ollama model${filteredOllama.length === 1 ? "" : "s"}`
                    : `${filtered.length} catalog · ${filteredOllama.length} Ollama`}
                </p>
              )}
              {progress ? (
                <div className="progress global-progress" aria-live="polite">
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${Math.max(2, Math.min(100, progress.pct))}%` }} />
                  </div>
                  <div className="progress-meta">
                    <span>{Math.round(progress.pct)}%</span>
                    <span>{progress.message}</span>
                  </div>
                </div>
              ) : null}

              {filteredOllama.length ? (
                <div className="ollama-block">
                  <div className="ollama-head">
                    <h2 className="ollama-title">Ollama (already on this machine)</h2>
                    <p className="ollama-note">
                      These run through your local Ollama server — not windhover-engine. No re-download.
                    </p>
                  </div>
                  <ul className="model-list">
                    {filteredOllama.map((m, idx) => (
                      <li
                        key={m.id}
                        className="model-row status-ollama"
                        style={{ animationDelay: `${Math.min(idx, 12) * 40}ms` }}
                      >
                        <div className="model-main">
                          <div className="model-title">
                            <h2>{(m.name || m.id).replace(/^ollama\//, "")}</h2>
                            <span className="badge ollama">Ollama</span>
                          </div>
                          <p>{m.description || "Preinstalled via Ollama"}</p>
                          <div className="meta">
                            <span>OLLAMA</span>
                            {m.size_bytes ? <span>{(m.size_bytes / 1e9).toFixed(1)} GB</span> : null}
                            <span>chat ready</span>
                          </div>
                        </div>
                        <div className="model-actions">
                          <button
                            type="button"
                            className="btn primary"
                            onClick={() => {
                              setActiveModel(m.id);
                              setTab("chat");
                            }}
                          >
                            Use in Chat
                          </button>
                          <button
                            type="button"
                            className="btn ghost"
                            onClick={() => {
                              setActiveModel(m.id);
                              setTab("agent");
                            }}
                          >
                            Use in Agent
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : family === "ollama" ? (
                <p className="status muted">
                  No Ollama models detected. Start Ollama (<code>ollama serve</code>) and pull a model
                  (<code>ollama pull …</code>), then refresh.
                </p>
              ) : null}

              {family !== "ollama" ? (
              <ul className="model-list">
                {filtered.map((m, idx) => {
                  const got = !!matchInstalled(installed, m.id);
                  const inst = matchInstalled(installed, m.id);
                  const badge = statusBadge(m);
                  const impostor = !!inst?.impostor && !inst?.incomplete;
                  const incomplete = !!inst?.incomplete;
                  const rowProgress = progress?.id === m.id ? progress : null;
                  const isBusy = busy === m.id;
                  const downloading = !!rowProgress || (isBusy && !!progress && progress.id === m.id);
                  return (
                    <li
                      key={m.id}
                      className={`model-row status-${badge.cls}${isBusy ? " is-busy" : ""}`}
                      style={{ animationDelay: `${Math.min(idx, 12) * 40}ms` }}
                    >
                      <div className="model-main">
                        <div className="model-title">
                          <h2>{m.name}</h2>
                          <span className={`badge ${badge.cls}`}>{badge.label}</span>
                          {impostor ? <span className="badge bad">Fake stub — remove</span> : null}
                          {incomplete && !downloading ? (
                            <span className="badge bad">Incomplete — reinstall</span>
                          ) : null}
                          {downloading ? <span className="badge ready">Downloading</span> : null}
                        </div>
                        <p>{m.description}</p>
                        <div className="meta">
                          <span>{(m.family || "other").toUpperCase()}</span>
                          <span>~{m.size_gb} GB</span>
                          {m.ram_gb ? <span>~{m.ram_gb}+ GB RAM</span> : null}
                          {m.license ? <span>{m.license}</span> : null}
                        </div>
                        {rowProgress ? (
                          <div className="progress" aria-live="polite">
                            <div className="progress-track">
                              <div
                                className="progress-fill"
                                style={{ width: `${Math.max(2, Math.min(100, rowProgress.pct))}%` }}
                              />
                            </div>
                            <div className="progress-meta">
                              <span>{Math.round(rowProgress.pct)}%</span>
                              <span>{rowProgress.message}</span>
                            </div>
                          </div>
                        ) : null}
                      </div>
                      <div className="model-actions">
                        {downloading ? (
                          <button type="button" className="btn primary" disabled>
                            {`Downloading… ${Math.round(rowProgress?.pct || 0)}%`}
                          </button>
                        ) : impostor ? (
                          <button
                            type="button"
                            className="btn ghost danger"
                            disabled={isBusy}
                            onClick={() => void uninstall(m.id, m.name, inst?.path ?? undefined)}
                          >
                            {isBusy ? "Removing…" : "Remove fake"}
                          </button>
                        ) : incomplete || !got ? (
                          <button
                            type="button"
                            className="btn primary"
                            disabled={isBusy || !!progress || m.status === "soon" || m.chat === "blocked"}
                            onClick={() => void pull(m, true)}
                          >
                            {m.status === "soon" || m.chat === "blocked"
                              ? "Not supported yet"
                              : incomplete
                              ? `Reinstall (~${m.size_gb} GB)`
                              : isMacSmall(m)
                                ? `Install (~${m.size_gb} GB)`
                                : m.status === "download"
                                  ? "Download weights"
                                  : "Install"}
                          </button>
                        ) : (
                          <>
                            {inst?.chat_ok ? (
                              <button type="button" className="btn primary" onClick={() => void openChat(m.id)}>
                                Chat
                              </button>
                            ) : (
                              <button type="button" className="btn ghost" disabled>
                                No chat yet
                              </button>
                            )}
                            <button
                              type="button"
                              className="btn ghost danger"
                              disabled={isBusy}
                              onClick={() => void uninstall(m.id, m.name, inst?.path ?? undefined)}
                            >
                              {isBusy ? "Removing…" : "Uninstall"}
                            </button>
                          </>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ul>
              ) : null}
            </section>
          </div>
        ) : null}

        {tab === "chat" ? (
          <section className="work-pane chat-work">
            <header className="session-bar">
              <div className="session-title">
                <h1>Chat</h1>
                <p>
                  {activeMeta
                    ? `${activeMeta.name || activeMeta.id}${
                        isOllamaModel(activeMeta)
                          ? " · Ollama"
                          : activeMeta.chat_mode === "preview"
                            ? " · transformers"
                            : ""
                      }`
                    : "Install a model from Library, or start Ollama"}
                </p>
              </div>
              <div className="live-stats" aria-live="polite" title="Live process + last reply">
                <span className={`live-pill ${engineState || "idle"}`}>
                  <i className="dot" aria-hidden />
                  {engineOk === false
                    ? "Offline"
                    : enginePresent === false
                      ? "No binary"
                      : lastStats?.engine_active === false
                        ? "Inactive"
                        : engineOk
                          ? "Engine"
                          : "…"}
                </span>
                <span className="live-pill">
                  <em>RAM</em>
                  {Number(stats?.rss_mb ?? lastStats?.rss_mb ?? 0).toFixed(0)}
                  <small>MB</small>
                </span>
                <span className="live-pill">
                  <em>Last</em>
                  {lastStats?.tokens_per_sec != null ? (
                    <>
                      {lastStats.tokens_per_sec}
                      <small>tok/s</small>
                    </>
                  ) : (
                    <>—</>
                  )}
                </span>
                <span className="live-pill">
                  <em>Latency</em>
                  {lastStats?.latency_ms != null ? (
                    <>
                      {lastStats.latency_ms}
                      <small>ms</small>
                    </>
                  ) : (
                    <>—</>
                  )}
                </span>
                {lastStats?.backend || sending ? (
                  <span className="live-pill muted-pill">
                    {sending ? "Generating…" : lastStats?.backend}
                  </span>
                ) : null}
              </div>
            </header>
            {engineInactiveBanner && tab === "chat" ? (
              <div className="engine-banner" role="alert">
                <strong>Engine inactive</strong>
                <span>{engineInactiveBanner}</span>
              </div>
            ) : null}
            <div className="thread work-scroll" ref={threadRef}>
              {messages.length === 0 && !sending ? (
                <div className="empty">
                  <strong>Ask locally.</strong>
                  <span>
                    Only real installs appear here. Large catalog models require a full download —
                    they never silently use another model.
                  </span>
                </div>
              ) : (
                <>
                  {messages.map((m, i) => (
                    <div className={`bubble ${m.role}`} key={i}>
                      {m.role === "assistant" ? (
                        <div className="md">
                          <MarkdownBody text={m.content} />
                        </div>
                      ) : (
                        m.content
                      )}
                      {m.role === "assistant" && m.stats?.tokens_per_sec ? (
                        <div className="bubble-meta">
                          {m.stats.tokens_per_sec} tok/s · {m.stats.latency_ms} ms
                          {m.stats.rss_mb ? ` · ${m.stats.rss_mb} MB RSS` : ""}
                          {m.stats.backend ? ` · ${m.stats.backend}` : ""}
                          {m.stats.preview_model ? ` · ${m.stats.preview_model}` : ""}
                        </div>
                      ) : null}
                    </div>
                  ))}
                  {sending ? (
                    <div className="bubble assistant thinking" aria-live="polite">
                      <span className="think-dots">
                        <i />
                        <i />
                        <i />
                      </span>
                      Windhover is thinking…
                    </div>
                  ) : null}
                </>
              )}
            </div>
            <div className="composer-dock">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Message Windhover…"
                rows={2}
                disabled={sending}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void send();
                  }
                }}
              />
              <div className="composer-footer">
                <label className="model-inline">
                  <span className="sr">Model</span>
                  {modelSelect}
                </label>
                <button
                  type="button"
                  className="btn primary"
                  disabled={sending || !chatCapable.length}
                  onClick={() => void send()}
                >
                  {sending ? "…" : "Send"}
                </button>
              </div>
            </div>
          </section>
        ) : null}

        {tab === "agent" ? (
          <section className="work-pane agent-work">
            <header className="session-bar">
              <div className="session-title">
                <h1>Agent</h1>
                <p>
                  {workspaceReady
                    ? workspace
                    : "Local LLM edits a folder you pick — list / read / write under that root only"}
                </p>
              </div>
            </header>

            <div className="agent-thread work-scroll">
              {agentSteps.length === 0 && !agentBusy && !agentPrompt ? (
                <div className="empty">
                  <strong>Ask the agent to change code.</strong>
                  <span>
                    Pick a folder in the sidebar and a capable model, then describe the edit.
                    Replies stay on-device.
                  </span>
                </div>
              ) : (
                <div className="agent-transcript">
                  {agentPrompt ? (
                    <div className="agent-user">
                      <span className="agent-role">You</span>
                      <p>{agentPrompt}</p>
                    </div>
                  ) : null}

                  {agentSteps.map((s) => {
                    const results = (s.tool_results || []) as AgentToolResult[];
                    const group = toolActivityGroup(results);
                    const hasTools = results.length > 0 || (s.tool_calls || []).length > 0;
                    const reply = agentVisibleReply(s.assistant || "", hasTools);
                    return (
                      <div className="agent-turn" key={s.step}>
                        {results.length ? (
                          <details className="agent-activity">
                            <summary>
                              {group || `${results.length} tool call${results.length === 1 ? "" : "s"}`}
                            </summary>
                            <ul>
                              {results.map((tr, i) => (
                                <li key={i} className={tr.ok === false ? "bad" : undefined}>
                                  {toolActivityLabel(tr)}
                                  {tr.ok === false && tr.error ? (
                                    <span className="agent-activity-err"> — {String(tr.error)}</span>
                                  ) : null}
                                </li>
                              ))}
                            </ul>
                          </details>
                        ) : null}
                        {reply ? (
                          <div className="agent-reply md">
                            <MarkdownBody text={reply} />
                          </div>
                        ) : null}
                      </div>
                    );
                  })}

                  {agentBusy ? (
                    <div className="agent-thinking" aria-live="polite">
                      <span className="think-dots">
                        <i />
                        <i />
                        <i />
                      </span>
                      <div className="agent-working">
                        <strong>{agentPhase === "tool" ? "Working in your files" : "Working"}</strong>
                        <span>{agentStatus || "Starting…"}</span>
                      </div>
                    </div>
                  ) : null}

                  {agentSummary && !agentBusy ? (
                    <p className="agent-footer muted">{agentSummary}</p>
                  ) : null}
                </div>
              )}
            </div>

            <div className="composer-dock">
              <textarea
                value={agentInput}
                onChange={(e) => setAgentInput(e.target.value)}
                placeholder="Ask the agent to explore or edit files…"
                rows={3}
                disabled={agentBusy}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void runAgent();
                  }
                }}
              />
              <div className="composer-footer">
                <label className="model-inline">
                  <span className="sr">Model</span>
                  {modelSelect}
                </label>
                <button
                  type="button"
                  className="btn primary"
                  disabled={agentBusy || !chatCapable.length}
                  onClick={() => void runAgent()}
                >
                  {agentBusy ? "…" : "Run"}
                </button>
              </div>
            </div>
          </section>
        ) : null}

        {tab === "advanced" ? (
          <section className="advanced-pane library-scroll">
            <header className="session-bar">
              <div className="session-title">
                <h1>Advanced</h1>
                <p>Live telemetry — no fake model routing</p>
              </div>
              <div className="session-actions">
                <button type="button" className="btn ghost" onClick={() => void refresh()}>
                  Refresh
                </button>
                <button
                  type="button"
                  className="btn ghost"
                  disabled={updateBusy}
                  onClick={() => void checkForUpdate(true)}
                >
                  Check for updates
                </button>
              </div>
            </header>

            <div className="adv-block update-block">
              <h2>App updates</h2>
              <p className="muted">
                Current version: <code>{appVersion || updateInfo?.current || "…"}</code>
                {updateInfo?.latest ? (
                  <>
                    {" "}
                    · Latest: <code>{updateInfo.latest}</code>
                  </>
                ) : null}
              </p>
              {updateInfo?.available ? (
                <p>
                  A newer build is available. Click Update now — Windows replaces the install in
                  place; on macOS open the DMG and drag to Applications.
                </p>
              ) : (
                <p className="muted">{updateMsg || "You're up to date (checked against GitHub Releases)."}</p>
              )}
              <div className="modal-actions" style={{ marginTop: "0.75rem" }}>
                {updateInfo?.available ? (
                  <button
                    type="button"
                    className="btn primary"
                    disabled={updateBusy}
                    onClick={() => void applyUpdate()}
                  >
                    {updateBusy ? "Downloading…" : "Update now"}
                  </button>
                ) : null}
                {updateInfo?.html_url ? (
                  <a className="btn ghost" href={updateInfo.html_url} target="_blank" rel="noreferrer">
                    Open releases
                  </a>
                ) : null}
              </div>
            </div>

            <div className="metrics-grid">
              <div className={`metric engine-status-metric ${engineOk && enginePresent !== false ? "on" : "off"}`}>
                <span className="metric-label">Windhover engine</span>
                <strong className="metric-value">
                  {engineOk === false ? "Off" : enginePresent === false ? "No binary" : engineOk ? "On" : "…"}
                </strong>
                <span className="metric-sub">
                  {engineOk === false
                    ? "API unreachable — run ./windhover app"
                    : enginePresent === false
                      ? "Build with ./windhover build"
                      : "API + windhover-engine ready"}
                </span>
              </div>
              <div className="metric">
                <span className="metric-label">Process RSS</span>
                <strong className="metric-value">
                  {Number(lastStats?.rss_mb ?? stats?.rss_mb ?? 0).toFixed(1)}
                  <small> MB</small>
                </strong>
              </div>
              <div className="metric">
                <span className="metric-label">Last latency</span>
                <strong className="metric-value">
                  {lastStats?.latency_ms ?? "—"}
                  <small> ms</small>
                </strong>
              </div>
              <div className="metric">
                <span className="metric-label">Output speed</span>
                <strong className="metric-value">
                  {lastStats?.tokens_per_sec ?? "—"}
                  <small> tok/s</small>
                </strong>
              </div>
            </div>

            <div className="adv-block">
              <h2>Routing</h2>
              <dl className="kv">
                <div>
                  <dt>Selected model</dt>
                  <dd>{lastStats?.selected_model || activeModel || "—"}</dd>
                </div>
                <div>
                  <dt>Backend</dt>
                  <dd>{lastStats?.backend || "—"}</dd>
                </div>
                <div>
                  <dt>Chat weights</dt>
                  <dd>{lastStats?.preview_model || stats?.chat_preview || "windhover-engine SNAP"}</dd>
                </div>
                <div>
                  <dt>Engine binary</dt>
                  <dd className="mono">{stats?.engine || "—"}</dd>
                </div>
                <div>
                  <dt>Chat-capable installs</dt>
                  <dd>{chatCapable.length}</dd>
                </div>
              </dl>
            </div>

            {lastStats?.windhover ? (
              <div className="adv-block">
                <h2>Windhover</h2>
                <dl className="kv">
                  <div>
                    <dt>Decode</dt>
                    <dd>
                      {lastStats.windhover.decode_tok_s != null
                        ? `${lastStats.windhover.decode_tok_s} tok/s`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Prefill</dt>
                    <dd>
                      {lastStats.windhover.prefill_tok_s != null
                        ? `${lastStats.windhover.prefill_tok_s} tok/s`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Working-set footprint</dt>
                    <dd>
                      {lastStats.windhover.footprint_gb != null
                        ? `${lastStats.windhover.footprint_gb} GB`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>FFN sparsity</dt>
                    <dd>
                      {lastStats.windhover.sparsity_pct != null
                        ? `${lastStats.windhover.sparsity_pct}%`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Bytes / token</dt>
                    <dd>
                      {lastStats.windhover.bytes_per_tok != null
                        ? `${(lastStats.windhover.bytes_per_tok / 1e9).toFixed(2)} GB`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>AU hot hit</dt>
                    <dd>
                      {lastStats.windhover.au_hit_pct != null
                        ? `${lastStats.windhover.au_hit_pct}%`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Forwards</dt>
                    <dd>{lastStats.windhover.forwards ?? "—"}</dd>
                  </div>
                </dl>
              </div>
            ) : null}
          </section>
        ) : null}
        </main>
      </div>
    </div>
  );
}
