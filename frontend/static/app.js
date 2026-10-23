
const chatForm = document.getElementById("chatForm");
const queryInput = document.getElementById("queryInput");
const sendButton = document.getElementById("sendButton");
const sendLabel = document.getElementById("sendLabel");
const chatArea = document.getElementById("chatArea");
const welcome = document.getElementById("welcome");
const connectionStatus = document.getElementById("connectionStatus");
const newChatButton = document.getElementById("newChat");

function addMessage(role, text, metadata = null, isError = false) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role === "user" ? "message-user" : "message-assistant"}`;

    if (role === "user") {
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.textContent = text;
        wrapper.appendChild(bubble);
    } else {
        const card = document.createElement("div");
        card.className = "answer-card";

        const label = document.createElement("div");
        label.className = "answer-label";
        label.textContent = isError ? "Request error" : "Knowledge AI";

        const answer = document.createElement("div");
        answer.className = `answer-text${isError ? " error-text" : ""}`;
        answer.textContent = text;

        card.append(label, answer);

        if (metadata && typeof metadata === "object") {
            const retrieval = metadata.retrieval;

            if (retrieval && typeof retrieval === "object") {
                const details = document.createElement("div");
                details.className = "answer-meta";

                const count = Number(retrieval.document_count);
                const requested = Number(retrieval.requested_k);

                const parts = [];

                if (Number.isFinite(count)) {
                    parts.push(`Retrieved documents: ${count}`);
                }

                if (Number.isFinite(requested)) {
                    parts.push(`Requested results: ${requested}`);
                }

                details.textContent = parts.join(" · ");

                if (parts.length > 0) {
                    card.appendChild(details);
                }
            }
        }

        wrapper.appendChild(card);
    }

    chatArea.appendChild(wrapper);
    chatArea.scrollTop = chatArea.scrollHeight;
    wrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });

    return wrapper;
}

function getAnswerText(result) {
    if (typeof result === "string") {
        return result;
    }

    if (!result || typeof result !== "object") {
        return "The service returned an unexpected response.";
    }

    const keys = ["answer", "output", "response", "result"];

    for (const key of keys) {
        if (typeof result[key] === "string" && result[key].trim()) {
            return result[key];
        }
    }

    return JSON.stringify(result, null, 2);
}

async function checkBackend() {
    const dot = document.querySelector(".sidebar-bottom .status-dot");

    try {
        const response = await fetch("/health");
        if (!response.ok) {
            throw new Error("Backend unavailable");
        }

        const data = await response.json();

        if (data.backend && data.backend.status === "ok") {
            connectionStatus.textContent = "API connected";
            dot.classList.add("online");
            dot.classList.remove("offline");
        } else {
            throw new Error("Backend unavailable");
        }
    } catch {
        connectionStatus.textContent = "API unavailable";
        dot.classList.add("offline");
        dot.classList.remove("online");
    }
}

async function askQuestion(question) {
    const query = question.trim();

    if (!query || sendButton.disabled) {
        return;
    }

    if (welcome && welcome.isConnected) {
        welcome.remove();
    }

    addMessage("user", query);
    queryInput.value = "";
    queryInput.style.height = "auto";

    sendButton.disabled = true;
    sendLabel.textContent = "Thinking...";

    const loading = document.createElement("div");
    loading.className = "message message-assistant";
    loading.innerHTML = `
        <div class="loading-indicator">
            <span></span><span></span><span></span>
            <div>Searching your knowledge base...</div>
        </div>
    `;
    chatArea.appendChild(loading);
    loading.scrollIntoView({ behavior: "smooth", block: "nearest" });

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
        });

        const result = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(
                typeof result.error === "string"
                    ? result.error
                    : typeof result.detail === "string"
                        ? result.detail
                        : "The request failed. Please try again."
            );
        }

        loading.remove();
        addMessage("assistant", getAnswerText(result), result);
    } catch (error) {
        loading.remove();
        addMessage(
            "assistant",
            error instanceof Error ? error.message : "An unexpected error occurred.",
            null,
            true
        );
    } finally {
        sendButton.disabled = false;
        sendLabel.textContent = "Ask AI";
        queryInput.focus();
    }
}

chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    askQuestion(queryInput.value);
});

queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});

queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = `${Math.min(queryInput.scrollHeight, 180)}px`;
});

document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
        askQuestion(button.dataset.query || "");
    });
});

newChatButton.addEventListener("click", () => {
    chatArea.replaceChildren();

    const welcomeCopy = welcome
        ? welcome.cloneNode(true)
        : null;

    if (welcomeCopy) {
        welcomeCopy.id = "welcome";
        welcomeCopy.style.display = "";
        chatArea.appendChild(welcomeCopy);

        welcomeCopy.querySelectorAll(".suggestion").forEach((button) => {
            button.addEventListener("click", () => {
                askQuestion(button.dataset.query || "");
            });
        });
    } else {
        window.location.reload();
    }

    queryInput.value = "";
    queryInput.focus();
});

checkBackend();
