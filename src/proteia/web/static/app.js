// SPDX-License-Identifier: Apache-2.0
// The page shell: keeps this launch's access token and talks to the local server.
"use strict";

const TOKEN_KEY = "proteia-token";

// The launcher puts the token in the URL fragment, which the browser never sends
// to a server. Keep it for this tab only and remove it from the address bar.
function takeToken() {
  const match = /^#token=([A-Za-z0-9_-]{32,128})$/.exec(window.location.hash);
  if (match) {
    sessionStorage.setItem(TOKEN_KEY, match[1]);
    history.replaceState(null, "", window.location.pathname + window.location.search);
  }
  return sessionStorage.getItem(TOKEN_KEY);
}

const token = takeToken();
const statusLine = document.getElementById("status");
const quitButton = document.getElementById("quit");

// Every request names the token in a header; nothing relies on cookies.
function api(path, options = {}) {
  const headers = new Headers(options.headers);
  headers.set("Authorization", `Bearer ${token}`);
  return fetch(path, { ...options, headers, cache: "no-store" });
}

const NEEDS_LAUNCH =
  "This page needs the link Proteia opens when it starts. Start Proteia again to open it.";

async function showStatus() {
  if (!token) {
    statusLine.textContent = NEEDS_LAUNCH;
    return;
  }
  try {
    const response = await api("/api/status");
    if (response.status === 401) {
      statusLine.textContent = NEEDS_LAUNCH;
      return;
    }
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    const status = await response.json();
    statusLine.textContent = `Proteia ${status.version} is running.`;
    quitButton.hidden = false;
  } catch (error) {
    statusLine.textContent = "Proteia is not responding. Start it again to reopen this page.";
  }
}

quitButton.addEventListener("click", async () => {
  quitButton.disabled = true;
  try {
    await api("/api/quit", { method: "POST" });
    statusLine.textContent = "Proteia has stopped. You can close this tab.";
    quitButton.hidden = true;
  } catch (error) {
    statusLine.textContent = "Proteia did not respond to Quit.";
    quitButton.disabled = false;
  }
});

showStatus();
