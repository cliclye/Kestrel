use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use tauri::Manager;
use url::Url;

struct Backend(Mutex<Option<Child>>);

fn repo_root() -> PathBuf {
    // desktop/src-tauri → desktop → repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

fn health_ok() -> bool {
    let mut stream = match TcpStream::connect_timeout(
        &"127.0.0.1:8000".parse().unwrap(),
        Duration::from_millis(200),
    ) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(400)));
    let _ = stream.write_all(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    let mut buf = [0u8; 256];
    let n = stream.read(&mut buf).unwrap_or(0);
    let body = String::from_utf8_lossy(&buf[..n]);
    body.contains("200") && body.contains("\"ok\"")
}

fn start_backend() -> Option<Child> {
    let root = repo_root();
    let cli = root.join("windhover");
    let cli = if cli.is_file() { cli } else { root.join("kestrel") };
    if !cli.is_file() {
        eprintln!("[windhover-desktop] missing CLI at {}", cli.display());
        return None;
    }
    // If something healthy is already on :8000 (e.g. `./windhover app`), reuse it.
    if health_ok() {
        eprintln!("[windhover-desktop] reusing existing server on :8000");
        return None;
    }
    // Prefer project venv (torch/transformers for chat preview).
    let venv_py = root.join("c/.venv/bin/python3");
    let python = if venv_py.is_file() {
        venv_py
    } else {
        PathBuf::from("python3")
    };
    let mut child = Command::new(&python)
        .arg(&cli)
        .arg("app")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg("8000")
        .current_dir(&root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    for _ in 0..50 {
        if health_ok() {
            return Some(child);
        }
        if let Ok(Some(status)) = child.try_wait() {
            eprintln!("[windhover-desktop] backend exited early: {status}");
            return None;
        }
        thread::sleep(Duration::from_millis(100));
    }
    eprintln!("[windhover-desktop] backend started (health still warming up)");
    Some(child)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let child = start_backend();
            app.manage(Backend(Mutex::new(child)));

            if let Some(win) = app.get_webview_window("main") {
                let _ = win.set_title("Windhover");
                // Prefer the live Windhover server (UI + API same-origin). Asset:// alone
                // often white-screens when absolute Vite paths / API fetches fail.
                let win2 = win.clone();
                thread::spawn(move || {
                    for _ in 0..60 {
                        if health_ok() {
                            let url = Url::parse("http://127.0.0.1:8000/").expect("static url");
                            if let Err(e) = win2.navigate(url) {
                                eprintln!("[windhover-desktop] navigate failed: {e}");
                                let _ = win2.eval(
                                    "window.location.replace('http://127.0.0.1:8000/')",
                                );
                            }
                            break;
                        }
                        thread::sleep(Duration::from_millis(100));
                    }
                });
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.try_state::<Backend>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(mut child) = guard.take() {
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to run the Windhover macOS application");
}
