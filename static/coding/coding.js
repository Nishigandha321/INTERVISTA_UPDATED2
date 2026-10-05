(() => {
  const api = async (url, options = {}) => {
    const response = await fetch(url, {
      credentials: "same-origin",
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    let data = {};
    try { data = await response.json(); } catch (_) {}
    if (!response.ok) {
      const error = new Error(data.detail || "The coding request could not be completed.");
      error.status = response.status;
      error.data = data;
      throw error;
    }
    return data;
  };

  const errorBox = document.getElementById("coding-error");
  const statusBox = document.getElementById("execution-status");
  const runButton = document.getElementById("run-code");
  const submitButton = document.getElementById("submit-code");
  const languageSelect = document.getElementById("language-select");
  const resultsBox = document.getElementById("test-results");
  const editorNode = document.getElementById("monaco-editor");
  const savedCode = new Map();
  let editor = null;
  let attempt = null;
  let activeQuestion = null;
  let currentLanguage = "cpp";
  let busy = false;

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }
  function clearError() { errorBox.hidden = true; errorBox.textContent = ""; }
  function setBusy(value) {
    busy = value;
    runButton.disabled = value;
    submitButton.disabled = value;
    languageSelect.disabled = value;
  }
  function text(id, value) { document.getElementById(id).textContent = value || ""; }
  function languageKey() { return `${activeQuestion.question.id}:${languageSelect.value}`; }

  function renderQuestion(payload) {
    activeQuestion = payload;
    const q = payload.question;
    text("question-title", q.title);
    text("question-position", `QUESTION ${payload.question_order} OF 2`);
    text("question-difficulty", q.difficulty);
    text("question-topic", q.topic);
    text("question-description", q.description);
    text("question-constraints", q.constraints);
    text("question-input-format", q.input_format);
    text("question-output-format", q.output_format);
    renderExamples(q.examples || []);
    text("attempt-progress", `Question ${payload.question_order} of 2`);
    const selectedLanguage = payload.language || "cpp";
    currentLanguage = selectedLanguage;
    languageSelect.value = selectedLanguage;
    text("execution-counter", `${payload.submission_count ?? 0} / ${payload.submission_limit ?? 2} submissions`);
    const code = payload.saved_source_code ?? (selectedLanguage === "cpp" ? q.starter_code_cpp : q.starter_code_python);
    if (editor) {
      editor.setValue(code);
      monaco.editor.setModelLanguage(editor.getModel(), selectedLanguage === "cpp" ? "cpp" : "python");
    }
    savedCode.set(languageKey(), code);
    renderResults(payload.saved_results || []);
    statusBox.textContent = "";
  }

  function renderResults(results) {
    if (!results || !results.length) {
      resultsBox.innerHTML = '<p class="muted-copy">Run your code to see per-test results.</p>';
      return;
    }
    resultsBox.replaceChildren(...results.map((item) => {
      const row = document.createElement("div");
      row.className = `test-result ${item.passed ? "passed" : "failed"}`;
      const title = document.createElement("strong");
      title.textContent = `Test ${item.test_number} — ${item.status}`;
      row.append(title);
      if (item.execution_time !== null && item.execution_time !== undefined) {
        const meta = document.createElement("span");
        meta.textContent = `${item.execution_time} s${item.memory === null || item.memory === undefined ? "" : ` · ${item.memory} KB`}`;
        row.append(meta);
      }
      if (item.status === "Wrong Answer") {
        const expected = document.createElement("pre");
        expected.textContent = `Expected:\n${item.expected_output ?? ""}`;
        row.append(expected);
        const output = document.createElement("pre");
        output.textContent = `Your Output:\n${item.stdout || "(no output)"}`;
        row.append(output);
        const reason = document.createElement("p");
        reason.textContent = "Reason: Output does not match the expected format/value.";
        row.append(reason);
      } else if (item.stdout) {
        const output = document.createElement("pre");
        output.textContent = `Your Output:\n${item.stdout}`;
        row.append(output);
      }
      if (item.stderr) {
        const error = document.createElement("pre");
        error.textContent = item.stderr;
        row.append(error);
      }
      if (item.compiler_warning) {
        const warning = document.createElement("pre");
        warning.textContent = `Compiler warning:\n${item.compiler_warning}`;
        row.append(warning);
      }
      return row;
    }));
  }

  function renderExamples(examples) {
    const container = document.getElementById("question-examples");
    if (!container) return;
    if (!examples.length) {
      container.textContent = "No example is available.";
      return;
    }
    container.replaceChildren(...examples.map((example, index) => {
      const item = document.createElement("div");
      item.className = "question-example";
      const heading = document.createElement("strong");
      heading.textContent = examples.length > 1 ? `Example ${index + 1}` : "Example";
      const inputLabel = document.createElement("span");
      inputLabel.textContent = "Input";
      const input = document.createElement("pre");
      input.textContent = example.input;
      const outputLabel = document.createElement("span");
      outputLabel.textContent = "Output";
      const output = document.createElement("pre");
      output.textContent = example.output;
      item.append(heading, inputLabel, input, outputLabel, output);
      return item;
    }));
  }

  function submissionBody() {
    return {
      attempt_id: attempt.attempt_id,
      question_id: activeQuestion.question.id,
      language: languageSelect.value,
      source_code: editor.getValue(),
      // No tests, expected outputs, or scores are sent by the browser.
    };
  }

  function updateExecutionState(data) {
    renderResults(data.results || []);
    text("execution-counter", `${data.submission_count ?? activeQuestion.submission_count ?? 0} / ${data.submission_limit ?? activeQuestion.submission_limit ?? 2} submissions`);
  }

  async function runCode() {
    if (busy || !activeQuestion) return;
    clearError();
    setBusy(true);
    statusBox.textContent = "Evaluating your code against the server-side test suite…";
    try {
      const data = await api("/api/coding/run", { method: "POST", body: JSON.stringify(submissionBody()) });
      updateExecutionState(data);
      statusBox.textContent = data.message || "Evaluation complete. Submit this result or revise your code for the remaining submission.";
    } catch (error) {
      if (error.data?.results) {
        updateExecutionState(error.data);
        const resultsCompleted = error.data.results.length === 3 && error.data.results.every(
          (item) => item.status && item.status !== "Not Run" && item.status !== "Execution Provider Error"
        );
        if (resultsCompleted && !error.data.incomplete) {
          statusBox.textContent = `Run completed. ${error.message}`;
          return;
        }
      }
      showError(error.message);
      statusBox.textContent = "Run did not complete.";
    } finally { setBusy(false); }
  }

  async function submitCode() {
    if (busy || !activeQuestion) return;
    clearError();
    setBusy(true);
    statusBox.textContent = "Submitting this question…";
    try {
      const data = await api("/api/coding/submit", { method: "POST", body: JSON.stringify(submissionBody()) });
      updateExecutionState(data);
      if (data.can_retry) {
        statusBox.textContent = data.message || "Some tests failed. Revise your code and use your second submission.";
        return;
      }
      if (data.completed) {
        statusBox.textContent = "Coding Round complete. Opening your report…";
        window.location.assign(data.result_url);
        return;
      }
      attempt = { ...attempt, ...data };
      renderQuestion(data.next);
      statusBox.textContent = "Question saved. Continue to Question 2.";
    } catch (error) {
      if (error.data?.results) updateExecutionState(error.data);
      showError(error.message);
      statusBox.textContent = "This question has not been finalized.";
    } finally { setBusy(false); }
  }

  function setEditorLanguage(nextLanguage) {
    if (!editor || !activeQuestion) return;
    const previousKey = `${activeQuestion.question.id}:${currentLanguage}`;
    savedCode.set(previousKey, editor.getValue());
    const nextKey = `${activeQuestion.question.id}:${nextLanguage}`;
    const starter = nextLanguage === "cpp" ? activeQuestion.question.starter_code_cpp : activeQuestion.question.starter_code_python;
    const nextCode = savedCode.has(nextKey) ? savedCode.get(nextKey) : starter;
    editor.setValue(nextCode);
    monaco.editor.setModelLanguage(editor.getModel(), nextLanguage === "cpp" ? "cpp" : "python");
    currentLanguage = nextLanguage;
    savedCode.set(nextKey, nextCode);
  }

  function initializeEditor() {
    if (typeof require !== "function") throw new Error("Monaco Editor could not be loaded. Check the network connection and retry.");
    require(["vs/editor/editor.main"], () => {
      editor = monaco.editor.create(editorNode, {
        value: "",
        language: "cpp",
        theme: "vs-dark",
        automaticLayout: true,
        minimap: { enabled: false },
        lineNumbers: "on",
        tabSize: 4,
        insertSpaces: true,
        fontSize: 14,
        scrollBeyondLastLine: false,
      });
      if (activeQuestion) renderQuestion(activeQuestion);
    }, () => showError("Monaco Editor could not be loaded. Check the network connection and retry."));
  }

  async function startAttempt() {
    attempt = await api("/api/coding/attempt/start", { method: "POST", body: "{}" });
    if (attempt.is_complete) {
      window.location.assign(`/coding/attempt/${attempt.attempt_id}/result`);
      return;
    }
    renderQuestion(attempt);
  }

  runButton.addEventListener("click", runCode);
  submitButton.addEventListener("click", submitCode);
  languageSelect.addEventListener("change", (event) => setEditorLanguage(event.target.value));
  window.addEventListener("beforeunload", () => {
    if (editor && activeQuestion) savedCode.set(languageKey(), editor.getValue());
  });
  initializeEditor();
  startAttempt().catch((error) => showError(error.message));
})();
