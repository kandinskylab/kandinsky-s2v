import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

export async function readEvents(response, onEvent) {
    if (!response.ok) throw new Error(await response.text());
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    let complete = false;
    try {
        while (true) {
            const { value, done } = await reader.read();
            pending += decoder.decode(value, { stream: !done });
            let newline;
            while ((newline = pending.indexOf("\n")) !== -1) {
                const line = pending.slice(0, newline);
                pending = pending.slice(newline + 1);
                if (!line.trim()) continue;
                const event = JSON.parse(line);
                if (event.status === "error") throw new Error(event.message);
                complete ||= event.status === "complete";
                onEvent(event);
            }
            if (done) break;
        }
        if (!complete) throw new Error("Connection interrupted. Retry to resume the download.");
    } finally {
        reader.releaseLock();
    }
}

function isModelsNote(node, packageId) {
    return node?.type === "MarkdownNote" && node.properties?.kandinsky_s2v_download_package === packageId;
}

export function registerDownloadUI(packageId, displayName) {
    let dialog = null;
    // workflow: the packaged workflow whose models to fetch. Undefined from the
    // menu: taken from the Models note of the open graph, else chosen in the dialog.
    const open = async (workflow) => {
        if (dialog?.isConnected) {
            if (!dialog.open) dialog.showModal();
            return;
        }
        dialog = document.createElement("dialog");
        Object.assign(dialog.style, {
            width: "min(640px, 90vw)", padding: "20px", borderRadius: "8px",
            color: "var(--input-text, #eee)", background: "var(--comfy-menu-bg, #222)",
            border: "1px solid var(--border-color, #666)",
        });
        const heading = document.createElement("h2");
        heading.textContent = displayName + " — Download models";
        const description = document.createElement("p");
        description.textContent = "Downloads weights and JSON configs to the ComfyUI server. "
            + "Large files: check free disk space. Existing files are reused. "
            + "Nothing is downloaded until you press Download.";
        const choice = document.createElement("select");
        choice.style.marginBottom = "12px";
        const status = document.createElement("pre");
        Object.assign(status.style, { whiteSpace: "pre-wrap", overflowWrap: "anywhere" });
        status.textContent = "Checking local models…";
        const download = document.createElement("button");
        download.textContent = "Download";
        download.disabled = true;
        const close = document.createElement("button");
        close.textContent = "Close";
        close.style.marginLeft = "12px";
        close.onclick = () => dialog.close();
        let busy = false;
        dialog.addEventListener("cancel", (event) => { if (busy) event.preventDefault(); });
        dialog.addEventListener("close", () => dialog.remove());
        dialog.append(heading, description, choice, status, download, close);
        document.body.append(dialog);
        dialog.showModal();
        try {
            const response = await api.fetchApi("/" + packageId + "/models");
            if (!response.ok) throw new Error("Restart ComfyUI to enable model downloads.");
            const { workflows } = await response.json();
            const names = Object.keys(workflows);
            workflow ??= (app.graph?.nodes ?? []).find((node) => isModelsNote(node, packageId))
                ?.properties.kandinsky_s2v_workflow;
            for (const name of names) choice.add(new Option(name, name, false, name === workflow));
            // One workflow known: no choice to offer.
            choice.style.display = names.includes(workflow) ? "none" : "";
            const show = () => {
                const models = workflows[choice.value];
                const ready = models.filter((model) => model.ready).length;
                status.textContent = choice.value + ": " + ready + " / " + models.length
                    + " files already available.\nDestination: ComfyUI models folders.";
            };
            choice.onchange = show;
            show();
            download.disabled = false;
        } catch (error) {
            status.textContent = error.message;
        }
        download.onclick = async () => {
            busy = true;
            download.disabled = close.disabled = choice.disabled = true;
            try {
                const response = await api.fetchApi("/" + packageId + "/download-models", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ confirm: true, workflow: choice.value }),
                });
                await readEvents(response, (event) => {
                    if (event.status === "complete") {
                        status.textContent = "All models are ready. Close this window and run the workflow.";
                    } else {
                        status.textContent = event.index + " / " + event.total + " — "
                            + event.status + "\n" + event.path;
                    }
                });
                await app.refreshComboInNodes();
            } catch (error) {
                status.textContent = error.message;
            } finally {
                busy = false;
                download.disabled = close.disabled = choice.disabled = false;
            }
        };
    };
    const command = packageId + ".download-models";
    app.registerExtension({
        name: packageId + ".downloads",
        commands: [{ id: command, label: displayName + " — Download models", function: () => open() }],
        menuCommands: [{ path: [displayName], commands: [command] }],
        loadedGraphNode(node) {
            if (!isModelsNote(node, packageId)
                || node.widgets?.some((widget) => widget.name === "Download models")) return;
            // A note that names no workflow (an older or hand-made graph) gets the choice in the dialog.
            const workflow = node.properties.kandinsky_s2v_workflow;
            // The right-side panel of the ComfyUI frontend (1.53) renders button widgets without
            // their callback, so a click there is lost; it does bind the widget options onto the
            // button, which makes onClick work there too. Where both fire, open() runs once.
            const onClick = () => open(workflow);
            node.addWidget("button", "Download models", null, onClick, { serialize: false, onClick });
        },
    });
}
