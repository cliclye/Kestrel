import { useEffect, useMemo, useRef, useState } from "react";
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
  path?: string;
  name?: string;
  ready?: boolean;
  engine?: string;
  family?: string;
  chat_ok?: boolean;
  chat_mode?: string;
  impostor?: boolean;
  size_bytes?: number;
  weight_bytes?: number;
  has_weights?: boolean;
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

const FAMILIES = [
  { id: "all", label: "All" },
  { id: "mac", label: "Mac 16GB" },
  { id: "windhover", label: "Windhover" },
  { id: "glm", label: "GLM" },
  { id: "qwen", label: "Qwen" },
  { id: "kimi", label: "Kimi" },
  { id: "deepseek", label: "DeepSeek" },
  { id: "mistral", label: "Mistral" },
  { id: "llama", label: "Llama" },
] as const;

function matchInstalled(list: Installed[], id: string) {
  const key = id.replace("/", "__");
  return list.find(
    (m) => m.id === id || m.id === key || m.id?.endsWith(id.split("/").pop() || "")
  );
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

function MarkdownBody({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        ),
      }}
    >
      {text}
    </ReactMarkdown>
  );
}

export function App() {
  const [tab, setTab] = useState<Tab>("library");
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
  const [tree, setTree] = useState<Array<{ name: string; path: string; type: string; size?: number | null }>>([]);
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

  async function refresh() {
    try {
      const health = await fetch(apiUrl("/health")).then((r) => r.json());
      setEngineOk(!!health?.ok);
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
      }
      if (health?.stats) setLastStats(health.stats);
      // Don't wipe download / uninstall status from the 5s poller
      if (!busyRef.current && !progressRef.current) setStatus("");
      setActiveModel((prev) => {
        const ok = list.filter((m) => m.chat_ok && !m.impostor);
        if (prev && ok.some((m) => m.id === prev || m.id === prev.replace("/", "__"))) return prev;
        const preview = ok.find((m) => String(m.id).includes("chat-preview"));
        return preview ? String(preview.id) : ok[0] ? String(ok[0].id) : "";
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

  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return catalog.filter((m) => {
      if (family === "mac") {
        if (!isMacSmall(m) && m.id !== "windhover/chat-preview") return false;
      } else if (family !== "all" && (m.family || "other") !== family) {
        return false;
      }
      if (!q) return true;
      const hay = `${m.name} ${m.id} ${m.description} ${(m.tags || []).join(" ")}`.toLowerCase();
      return hay.includes(q);
    });
  }, [catalog, family, query]);

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
            if (m.chat === "preview" || isMacSmall(m) || m.id.includes("chat-preview")) {
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
        if (m.chat === "preview" || isMacSmall(m) || m.id.includes("chat-preview")) {
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
      setStatus("That pack can’t chat — install Windhover Chat Preview, or download real weights.");
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
      setStatus("Install Windhover Chat Preview from Library first");
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
          max_tokens: 256,
          temperature: 0.7,
        }),
      });
      const j = await r.json();
      const content =
        j?.choices?.[0]?.message?.content ||
        j?.error ||
        "No response.";
      const st = (j?.stats || {}) as ChatStats;
      setLastStats(st);
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
    } finally {
      setSending(false);
    }
  }

  async function applyWorkspace(path?: string) {
    const root = (path ?? workspace).trim();
    if (!root) {
      setStatus("Enter a folder path on this Mac");
      return;
    }
    try {
      const r = await fetch(apiUrl("/api/workspace"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ root }),
      });
      const j = await r.json();
      if (!j.ok) {
        setStatus(j.error || "Could not set workspace");
        return;
      }
      setWorkspace(String(j.root));
      setStatus(`Workspace: ${j.root}`);
      const treeR = await fetch(apiUrl("/api/workspace/tree?path=.")).then((x) => x.json());
      if (treeR.ok) setTree(treeR.entries || []);
      else setTree([]);
    } catch (e) {
      setStatus(String(e));
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
        }),
      });
      const j = await r.json();
      if (!j.ok) {
        setStatus(j.error || "Agent failed");
        setAgentSummary(String(j.error || "failed"));
        return;
      }
      setAgentSteps((j.steps || []) as AgentStep[]);
      setAgentSummary(String(j.summary || "done"));
      setStatus(`Agent finished · ${j.step_count || 0} step(s)`);
      const treeR = await fetch(apiUrl("/api/workspace/tree?path=.")).then((x) => x.json());
      if (treeR.ok) setTree(treeR.entries || []);
    } catch (e) {
      setStatus(String(e));
      setAgentSummary(String(e));
    } finally {
      setAgentBusy(false);
    }
  }

  return (
    <div className="shell">
      <div className="sky" aria-hidden />
      <div className="grain" aria-hidden />

      {confirmUninstall ? (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal">
            <h2>Uninstall model?</h2>
            <p>
              Remove <strong>{confirmUninstall.name}</strong> from this Mac? This deletes local files
              under <code>~/.windhover/models</code>.
            </p>
            <div className="modal-actions">
              <button type="button" className="btn ghost" onClick={() => setConfirmUninstall(null)}>
                Cancel
              </button>
              <button type="button" className="btn primary danger-fill" onClick={() => void confirmUninstallNow()}>
                Uninstall
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <header className="top titlebar">
        <div className="traffic-spacer" aria-hidden />
        <div className="brand-row">
          <img className="mark" src="./kestrel-icon.png" alt="" width={36} height={36} />
          <div className="brand">
            <strong>Windhover</strong>
            <span>windhover-engine</span>
          </div>
        </div>
        <div className="top-right">
          <div
            className={`engine-pill ${
              engineOk === false ? "off" : enginePresent === false ? "warn" : engineOk ? "on" : ""
            }`}
            title={
              engineOk === false
                ? "Windhover API unreachable"
                : enginePresent === false
                  ? "API up, but windhover-engine binary missing — run ./windhover build"
                  : engineOk
                    ? "Windhover API online · windhover-engine ready"
                    : "Checking…"
            }
          >
            <i className="dot" aria-hidden />
            <div className="engine-pill-text">
              <span className="engine-pill-label">Windhover engine</span>
              <strong>
                {engineOk === false
                  ? "Off"
                  : enginePresent === false
                    ? "Binary missing"
                    : engineOk
                      ? "On"
                      : "…"}
              </strong>
            </div>
          </div>
          <nav className="nav">
            <button type="button" className={tab === "library" ? "active" : ""} onClick={() => setTab("library")}>
              Library
            </button>
            <button type="button" className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>
              Chat
            </button>
            <button type="button" className={tab === "agent" ? "active" : ""} onClick={() => setTab("agent")}>
              Agent
            </button>
            <button type="button" className={tab === "advanced" ? "active" : ""} onClick={() => setTab("advanced")}>
              Advanced
            </button>
          </nav>
        </div>
      </header>

      <main className="main">
        {tab === "library" ? (
          <>
            <section className="hero">
              <p className="brand-hero">Windhover</p>
              <h1>Honest local MoE.</h1>
              <p className="lead">
                Frontier models need a real download. Chat Preview is a small on-device model —
                never a stand-in for Kimi or Qwen.
              </p>
              <div className="hero-cta">
                <button
                  type="button"
                  className="btn primary"
                  onClick={() => {
                    const prev = catalog.find((m) => m.id === "windhover/chat-preview");
                    if (prev) void pull(prev);
                  }}
                >
                  Install Chat Preview
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

              {status ? <p className="status">{status}</p> : <p className="status muted">{filtered.length} models</p>}
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

              <ul className="model-list">
                {filtered.map((m, idx) => {
                  const got = !!matchInstalled(installed, m.id);
                  const inst = matchInstalled(installed, m.id);
                  const badge = statusBadge(m);
                  const impostor = !!inst?.impostor;
                  const rowProgress = progress?.id === m.id ? progress : null;
                  const isBusy = busy === m.id;
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
                        {impostor ? (
                          <button
                            type="button"
                            className="btn ghost danger"
                            disabled={isBusy}
                            onClick={() => void uninstall(m.id, m.name, inst?.path)}
                          >
                            {isBusy ? "Removing…" : "Remove fake"}
                          </button>
                        ) : !got ? (
                          <button
                            type="button"
                            className="btn primary"
                            disabled={isBusy || !!progress}
                            onClick={() => void pull(m, isMacSmall(m))}
                          >
                            {isBusy
                              ? `Downloading… ${Math.round(rowProgress?.pct || 0)}%`
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
                              onClick={() => void uninstall(m.id, m.name, inst?.path)}
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
            </section>
          </>
        ) : null}

        {tab === "chat" ? (
          <section className="chat-pane">
            <header className="chat-head">
              <div>
                <h1>Chat</h1>
                <p>
                  {activeMeta
                    ? `Using ${activeMeta.name || activeMeta.id}${
                        activeMeta.chat_mode === "preview"
                          ? " · honest SmolLM2 preview (not a frontier MoE)"
                          : activeMeta.chat_mode === "engine-oracle"
                            ? " · engine oracle demo"
                            : ""
                      }`
                    : "Install Windhover Chat Preview from Library"}
                </p>
              </div>
              <label className="model-pick">
                <span>Model</span>
                <select
                  value={activeModel}
                  onChange={(e) => {
                    setActiveModel(e.target.value);
                    setMessages([]);
                  }}
                  disabled={!chatCapable.length}
                >
                  {!chatCapable.length ? <option value="">Install Chat Preview</option> : null}
                  {chatCapable.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name || m.id}
                    </option>
                  ))}
                </select>
              </label>
            </header>
            <div className="thread" ref={threadRef}>
              {messages.length === 0 && !sending ? (
                <div className="empty">
                  <strong>Ask locally.</strong>
                  <span>
                    Only real installs appear here. Kimi/Qwen/etc. require a full download — they
                    never silently use another model.
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
            <div className="composer">
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
              <button
                type="button"
                className="btn primary"
                disabled={sending || !chatCapable.length}
                onClick={() => void send()}
              >
                {sending ? "…" : "Send"}
              </button>
            </div>
          </section>
        ) : null}

        {tab === "agent" ? (
          <section className="chat-pane agent-pane">
            <header className="chat-head">
              <div>
                <h1>Agent</h1>
                <p>Local LLM edits a folder you pick — list / read / write under that root only</p>
              </div>
              <label className="model-pick">
                <span>Model</span>
                <select
                  value={activeModel}
                  onChange={(e) => setActiveModel(e.target.value)}
                  disabled={!chatCapable.length}
                >
                  {!chatCapable.length ? <option value="">Install a model</option> : null}
                  {chatCapable.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name || m.id}
                    </option>
                  ))}
                </select>
              </label>
            </header>

            <div className="agent-workspace">
              <label className="workspace-field">
                <span>Folder</span>
                <input
                  value={workspace}
                  onChange={(e) => setWorkspace(e.target.value)}
                  placeholder="/path/to/your/project"
                  disabled={agentBusy}
                />
              </label>
              <button type="button" className="btn ghost" disabled={agentBusy} onClick={() => void applyWorkspace()}>
                Use folder
              </button>
            </div>

            <div className="agent-body">
              <aside className="agent-tree" aria-label="Workspace files">
                <strong>Files</strong>
                {tree.length === 0 ? (
                  <p className="muted">Set a folder to list files</p>
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
              </aside>
              <div className="agent-thread">
                {agentSteps.length === 0 && !agentBusy ? (
                  <div className="empty">
                    <strong>Prompt the local agent.</strong>
                    <span>
                      Pick a project folder, choose a model (Coder 1.5B works well on 16GB), then ask it to
                      change files — like a lightweight Cursor, fully on-device.
                    </span>
                  </div>
                ) : (
                  <>
                    {agentSteps.map((s) => (
                      <div className="agent-step" key={s.step}>
                        <div className="agent-step-head">Step {s.step}</div>
                        {s.tool_results?.length ? (
                          <ul className="tool-log">
                            {s.tool_results.map((tr, i) => (
                              <li key={i}>
                                <code>{String((tr as { tool?: string }).tool || "tool")}</code>
                                {(tr as { ok?: boolean }).ok === false
                                  ? ` · ${(tr as { error?: string }).error || "failed"}`
                                  : (tr as { path?: string }).path
                                    ? ` · ${(tr as { path?: string }).path}`
                                    : (tr as { summary?: string }).summary
                                      ? ` · ${(tr as { summary?: string }).summary}`
                                      : " · ok"}
                              </li>
                            ))}
                          </ul>
                        ) : null}
                        {s.assistant ? (
                          <div className="bubble assistant md">
                            <MarkdownBody text={s.assistant.slice(0, 1200)} />
                          </div>
                        ) : null}
                      </div>
                    ))}
                    {agentBusy ? (
                      <div className="bubble assistant thinking" aria-live="polite">
                        <span className="think-dots">
                          <i />
                          <i />
                          <i />
                        </span>
                        Agent working in your folder…
                      </div>
                    ) : null}
                    {agentSummary && !agentBusy ? (
                      <div className="bubble assistant">
                        <strong>Done.</strong> {agentSummary}
                      </div>
                    ) : null}
                  </>
                )}
              </div>
            </div>

            <div className="composer">
              <textarea
                value={agentInput}
                onChange={(e) => setAgentInput(e.target.value)}
                placeholder="e.g. Add a hello() function to main.py and a one-line docstring"
                rows={2}
                disabled={agentBusy}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void runAgent();
                  }
                }}
              />
              <button
                type="button"
                className="btn primary"
                disabled={agentBusy || !chatCapable.length}
                onClick={() => void runAgent()}
              >
                {agentBusy ? "…" : "Run"}
              </button>
            </div>
          </section>
        ) : null}

        {tab === "advanced" ? (
          <section className="advanced-pane">
            <header className="chat-head">
              <div>
                <h1>Advanced</h1>
                <p>Live telemetry — no fake model routing</p>
              </div>
              <button type="button" className="btn ghost" onClick={() => void refresh()}>
                Refresh
              </button>
            </header>

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
  );
}
