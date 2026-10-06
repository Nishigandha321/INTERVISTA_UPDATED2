(() => {
  const form = document.querySelector(".gd-form");
  if (form) {
    const topicInput = form.querySelector('[name="custom_topic"]');
    const syncTopicField = () => {
      const custom = form.querySelector('[name="topic_mode"]:checked')?.value === "custom";
      topicInput.required = custom;
      topicInput.disabled = !custom;
    };
    form.querySelectorAll('[name="topic_mode"]').forEach((input) => input.addEventListener("change", syncTopicField));
    syncTopicField();
  }

  const root = document.getElementById("gd-session");
  if (!root) return;
  const publicId = root.dataset.publicId;
  const historyEl = document.getElementById("gd-history");
  const phaseEl = document.getElementById("gd-phase");
  const roundEl = document.getElementById("gd-round");
  const topicEl = document.getElementById("gd-topic");
  const statusEl = document.getElementById("gd-status");
  const participantsEl = document.getElementById("gd-participants");
  const elapsedEl = document.getElementById("gd-elapsed");
  const timeLeftEl = document.getElementById("gd-time-left");
  const progressFill = document.getElementById("gd-progress-fill");
  const warningEl = document.getElementById("gd-warning");
  const loadingEl = document.getElementById("gd-loading");
  const userTurnEl = document.getElementById("gd-user-turn");
  const completeEl = document.getElementById("gd-complete");
  const transcriptEl = document.getElementById("gd-transcript");
  const errorEl = document.getElementById("gd-error");
  const submitEl = document.getElementById("gd-submit");
  const recordEl = document.getElementById("gd-record");
  const recordStateEl = document.getElementById("gd-record-state");
  const userTimeLeftEl = document.getElementById("gd-user-time-left");
  const expiredEl = document.getElementById("gd-expired");
  let state = JSON.parse(root.dataset.state || "{}");
  let busy = false;
  let mediaRecorder = null;
  let audioChunks = [];
  let stream = null;
  let speakingId = null;
  let pendingSpeech = null;
  let recordedDurationSeconds = 0;
  let recordingStartedAt = 0;
  let recordingTurnToken = null;
  let discardRecording = false;
  let polling = false;

  const initials = (name) => name === "You" ? "YOU" : name.split(/\s+/).map((part) => part[0]).join("").slice(0, 2).toUpperCase();
  const phaseLabel = (phase) => ({opening:"Opening round",main_discussion:"Main discussion",conclusion:"Final round",completed:"Complete",expired:"Session ended"}[phase] || phase);
  const escapeText = (value) => String(value ?? "");

  function render() {
    topicEl.textContent = state.topic;
    phaseEl.textContent = phaseLabel(state.phase);
    roundEl.textContent = state.phase === "completed" ? "Session ended" : `Round ${state.round}`;
    warningEl.hidden = !state.final_round_warning;
    historyEl.replaceChildren();
    let lastPhase = "";
    (state.discussion_history || []).forEach((entry) => {
      if (entry.phase !== lastPhase) {
        const phase = document.createElement("div");
        phase.className = "gd-entry-phase";
        phase.textContent = `${phaseLabel(entry.phase)} · Round ${entry.round}`;
        historyEl.append(phase);
        lastPhase = entry.phase;
      }
      const row = document.createElement("article");
      row.className = "gd-entry";
      row.dataset.persona = entry.speaker === "You" ? "You" : (entry.persona || "");
      const avatar = document.createElement("span");
      avatar.className = "gd-entry-avatar";
      avatar.textContent = initials(entry.speaker);
      const body = document.createElement("div");
      const meta = document.createElement("div");
      meta.className = "gd-entry-meta";
      const name = document.createElement("strong");
      name.textContent = entry.speaker;
      const persona = document.createElement("span");
      persona.textContent = entry.speaker === "You" ? "CANDIDATE" : (entry.persona || "AI PARTICIPANT").toUpperCase();
      const speech = document.createElement("p");
      speech.textContent = escapeText(entry.content);
      meta.append(name, persona);
      body.append(meta, speech);
      row.append(avatar, body);
      historyEl.append(row);
    });
    participantsEl.replaceChildren();
    (state.participants || []).forEach((person) => {
      const row = document.createElement("div");
      const active = speakingId ? speakingId === person.id : state.current_speaker?.id === person.id;
      row.className = `gd-participant${active ? " active" : ""}`;
      row.dataset.persona = person.name;
      const avatar = document.createElement("span");
      avatar.className = "gd-participant-avatar";
      avatar.textContent = initials(person.name);
      const name = document.createElement("span");
      name.className = "gd-participant-name";
      name.textContent = person.name;
      const tag = document.createElement("span");
      tag.className = "gd-participant-tag";
      tag.textContent = speakingId === person.id ? "SPEAKING" : (state.current_speaker?.id === person.id ? (person.id === "user" ? "YOUR TURN" : "UP NEXT") : (person.id === "user" ? "YOU" : "AI"));
      row.append(avatar, name, tag);
      participantsEl.append(row);
    });
    completeEl.hidden = state.phase !== "completed";
    expiredEl.hidden = state.phase !== "expired";
    document.getElementById("gd-report-link").href = `/gd/report/${publicId}`;
    userTurnEl.hidden = ["completed", "expired"].includes(state.phase) || state.current_speaker?.id !== "user";
    loadingEl.hidden = true;
    const activePerson = (state.participants || []).find((person) => person.id === (speakingId || state.current_speaker?.id));
    statusEl.lastChild.textContent = ["completed", "expired"].includes(state.phase) ? " Complete" : speakingId ? ` ${activePerson?.name || "AI"} is speaking` : state.current_speaker?.id === "user" ? " Your turn" : ` ${activePerson?.name || "Ready"} is next`;
    statusEl.querySelector("i").style.background = (speakingId || state.current_speaker?.id) === "user" ? "#f0643a" : "#91a393";
    updateClock();
  }

  function formatTime(seconds) {
    const safe = Math.max(0, Math.floor(seconds));
    return `${String(Math.floor(safe / 60)).padStart(2, "0")}:${String(safe % 60).padStart(2, "0")}`;
  }

  function updateClock() {
    const started = Date.parse(state.started_at || "");
    elapsedEl.textContent = `${formatTime(started ? (Date.now() - started) / 1000 : 0)} elapsed`;
    const secondsLeft = state.total_seconds_remaining == null
      ? Math.max(0, 300 - (started ? (Date.now() - started) / 1000 : 0))
      : Math.max(0, state.total_seconds_remaining - (Date.now() - (state._stateReceivedAt || Date.now())) / 1000);
    if (timeLeftEl) timeLeftEl.textContent = formatTime(secondsLeft);
    if (userTimeLeftEl) userTimeLeftEl.textContent = formatTime(state.turn_seconds_remaining == null ? 0 : Math.max(0, state.turn_seconds_remaining - (Date.now() - (state._stateReceivedAt || Date.now())) / 1000));
    if (state.phase === "main_discussion") {
      const total = (state.participants || []).length || 1;
      const done = (state.discussion_history || []).filter((entry) => entry.phase === "main_discussion").length;
      progressFill.style.width = `${15 + 70 * done / total}%`;
    } else if (state.phase === "opening") {
      const total = (state.participants || []).length || 1;
      const done = (state.discussion_history || []).filter((entry) => entry.phase === "opening").length;
      progressFill.style.width = `${Math.min(15, 15 * done / total)}%`;
    } else if (state.phase === "conclusion") {
      const total = (state.participants || []).length || 1;
      const done = (state.discussion_history || []).filter((entry) => entry.phase === "conclusion").length;
      if (timeLeftEl) timeLeftEl.textContent = formatTime(secondsLeft);
      progressFill.style.width = `${85 + 15 * done / total}%`;
    } else {
      if (timeLeftEl) timeLeftEl.textContent = state.phase === "expired" ? "Ended" : "Complete";
      progressFill.style.width = "100%";
    }
  }

  function speakAiEntry(entry) {
    if (!("speechSynthesis" in window) || !window.SpeechSynthesisUtterance) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const utterance = new SpeechSynthesisUtterance(entry.content);
      const available = window.speechSynthesis.getVoices().filter((voice) => /^en([-_]|$)/i.test(voice.lang || ""));
      const personaIndex = Math.max(0, Number(String(entry.speaker_id || "ai_1").replace("ai_", "")) - 1);
      if (available.length) utterance.voice = available[personaIndex % available.length];
      utterance.rate = [0.96, 0.91, 1.0, 1.04][personaIndex % 4];
      utterance.pitch = [1.04, 0.91, 1.0, 1.08][personaIndex % 4];
      utterance.onend = resolve;
      utterance.onerror = (event) => reject(new Error(event.error || "Speech playback stopped before it finished"));
      window.speechSynthesis.speak(utterance);
    });
  }

  function offerSpeechRetry(entry) {
    let retry = document.getElementById("gd-speech-retry");
    if (!retry) {
      retry = document.createElement("button");
      retry.id = "gd-speech-retry";
      retry.className = "gd-retry";
      retry.type = "button";
      retry.textContent = "Replay participant response";
      loadingEl.after(retry);
    }
    retry.onclick = async () => {
      if (busy || !pendingSpeech) return;
      busy = true;
      retry.disabled = true;
      speakingId = entry.speaker_id;
      render();
      try {
        await speakAiEntry(entry);
        pendingSpeech = null;
        retry.remove();
        speakingId = null;
        render();
        busy = false;
        if (state.current_speaker?.id !== "user" && !["completed", "expired"].includes(state.phase)) runTurns();
      } catch (_) {
        retry.disabled = false;
        statusEl.lastChild.textContent = " Speech stopped early — replay the full response to continue";
        busy = false;
      }
    };
  }

  async function post(path, payload) {
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw Object.assign(new Error(result.detail || "The request could not be completed"), { status: response.status });
    result._stateReceivedAt = Date.now();
    return result;
  }

  async function refreshState() {
    const response = await fetch(`/api/gd/session/${publicId}`);
    if (!response.ok) throw new Error("Could not reload this discussion");
    state = await response.json();
    state._stateReceivedAt = Date.now();
    render();
  }

  async function pollState() {
    if (polling || busy) return;
    polling = true;
    try {
      const oldToken = state.turn_token;
      const oldSpeaker = state.current_speaker?.id;
      await refreshState();
      if (oldSpeaker === "user" && (state.current_speaker?.id !== "user" || state.turn_token !== oldToken) && mediaRecorder?.state === "recording") {
        discardRecording = true;
        mediaRecorder.stop();
      }
      if (state.current_speaker?.id !== "user" && !["completed", "expired"].includes(state.phase)) runTurns();
    } catch (_) { /* A later poll will retry a transient state read failure. */ }
    finally { polling = false; }
  }

  async function runTurns() {
    if (busy || pendingSpeech || ["completed", "expired"].includes(state.phase) || state.current_speaker?.id === "user") return;
    busy = true;
    speakingId = state.current_speaker?.id || null;
    render();
    loadingEl.hidden = false;
    statusEl.lastChild.textContent = ` ${state.current_speaker?.name || "Participant"} is preparing to speak`;
    const token = state.turn_token;
    try {
      state = await post(`/api/gd/session/${publicId}/ai-turn`, { turn_token: token });
      const entry = state.discussion_history[state.discussion_history.length - 1];
      speakingId = entry?.speaker_id || null;
      render();
      if (entry?.speaker_id?.startsWith("ai_")) {
        pendingSpeech = entry;
        try {
          await speakAiEntry(entry);
          pendingSpeech = null;
        } catch (_) {
          loadingEl.hidden = true;
          statusEl.lastChild.textContent = " Speech stopped early — replay the full response to continue";
          offerSpeechRetry(entry);
          busy = false;
          return;
        }
      }
      speakingId = null;
      render();
      if (state.current_speaker?.id !== "user" && !["completed", "expired"].includes(state.phase)) {
        busy = false;
        runTurns();
        return;
      }
    } catch (error) {
      if (error.status === 409) {
        await refreshState();
        if (state.current_speaker?.id !== "user" && !["completed", "expired"].includes(state.phase)) {
          busy = false;
          runTurns();
          return;
        }
      } else {
        speakingId = null;
        loadingEl.hidden = true;
        render();
        statusEl.lastChild.textContent = " Participant unavailable";
        let retry = document.getElementById("gd-retry");
        if (!retry) {
          retry = document.createElement("button");
          retry.id = "gd-retry";
          retry.className = "gd-retry";
          retry.type = "button";
          retry.textContent = "Retry participant turn";
          retry.addEventListener("click", () => { retry.remove(); runTurns(); });
          loadingEl.after(retry);
        }
      }
    }
    busy = false;
  }

  submitEl.addEventListener("click", async () => {
    if (busy) return;
    const transcript = transcriptEl.value.trim();
    if (!transcript) { errorEl.textContent = "Add a few words before continuing."; return; }
    busy = true;
    submitEl.disabled = true;
    errorEl.textContent = "";
    if (state.phase === "conclusion") statusEl.lastChild.textContent = " Saving final turn and preparing your report";
    let continueTurns = false;
    try {
      state = await post(`/api/gd/session/${publicId}/respond`, {
        transcript, turn_token: state.turn_token,
        input_mode: recordedDurationSeconds > 0 ? "voice" : "text",
        duration_seconds: recordedDurationSeconds || null,
      });
      recordedDurationSeconds = 0;
      transcriptEl.value = "";
      render();
      continueTurns = state.phase !== "completed" && state.current_speaker?.id !== "user";
    } catch (error) {
      errorEl.textContent = error.message;
      if (error.status === 409) await refreshState();
    } finally {
      busy = false;
      submitEl.disabled = false;
      if (continueTurns) runTurns();
    }
  });

  recordEl.addEventListener("click", async () => {
    errorEl.textContent = "";
    if (mediaRecorder?.state === "recording") { mediaRecorder.stop(); recordEl.classList.remove("is-recording"); recordEl.innerHTML = "<span></span>Start recording"; return; }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { errorEl.textContent = "Voice recording is not supported in this browser. You can type your contribution instead."; return; }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      recordingTurnToken = state.turn_token;
      discardRecording = false;
      mediaRecorder.ondataavailable = (event) => { if (event.data.size) audioChunks.push(event.data); };
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        recordEl.classList.remove("is-recording");
        recordEl.innerHTML = "<span></span>Start recording";
        if (discardRecording || state.current_speaker?.id !== "user" || state.turn_token !== recordingTurnToken) {
          audioChunks = [];
          recordStateEl.textContent = "That turn has ended. Wait for your next turn.";
          return;
        }
        recordedDurationSeconds += Math.max(1, (Date.now() - recordingStartedAt) / 1000);
        const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
        recordStateEl.textContent = "Transcribing your contribution…";
        recordEl.disabled = true;
        try {
          const formData = new FormData();
          formData.append("file", audioBlob, "gd-contribution.webm");
          formData.append("context_prompt", state.topic);
          const response = await fetch("/api/gd/transcribe", { method: "POST", body: formData });
          const result = await response.json();
          if (!response.ok) throw new Error(result.detail || "Transcription failed");
          transcriptEl.value = [transcriptEl.value.trim(), result.transcript].filter(Boolean).join(" ");
          recordStateEl.textContent = "Transcript added — review it, then submit.";
        } catch (error) { errorEl.textContent = error.message; recordStateEl.textContent = "You can enter your contribution by text."; }
        finally { recordEl.disabled = false; }
      };
      mediaRecorder.start();
      const activeRecorder = mediaRecorder;
      recordingStartedAt = Date.now();
      recordEl.classList.add("is-recording");
      recordEl.innerHTML = "<span></span>Stop recording";
      recordStateEl.textContent = "Recording — limited to 20 seconds so the discussion stays on time.";
      window.setTimeout(() => {
        if (mediaRecorder === activeRecorder && activeRecorder.state === "recording") activeRecorder.stop();
      }, 20000);
    } catch (_) { errorEl.textContent = "Microphone access was unavailable. You can type your contribution instead."; }
  });

  state._stateReceivedAt = Date.now();
  render();
  window.setInterval(updateClock, 1000);
  window.setInterval(pollState, 1000);
  runTurns();
})();
