const recordButton = document.querySelector('#record');
const upload = document.querySelector('#upload');
const analyze = document.querySelector('#analyze');
const preview = document.querySelector('#preview');
const message = document.querySelector('#message');
const timer = document.querySelector('#timer');
const result = document.querySelector('#result');
const displayLabels = {
  arabic: 'Arabic-influenced English',
  english: 'English first-language group',
  spanish: 'Spanish-influenced English',
  mandarin: 'Mandarin-influenced English',
  dutch: 'Dutch-influenced English',
};
let selectedFile = null, mediaRecorder = null, stream = null, ticker = null, started = 0, previewUrl = null;

if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
  recordButton.disabled = true;
  message.textContent = 'Microphone recording needs HTTPS or localhost in a supported browser. You can still upload audio.';
}

function showFile(file) {
  selectedFile = file;
  analyze.disabled = false;
  result.hidden = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  message.textContent = `${file.name} selected (${(file.size / 1024 / 1024).toFixed(2)} MB).`;
}

upload.addEventListener('change', () => { if (upload.files[0]) showFile(upload.files[0]); });

recordButton.addEventListener('click', async () => {
  if (mediaRecorder?.state === 'recording') { mediaRecorder.stop(); return; }
  message.textContent = 'Allow microphone access in your browser to start recording…';
  recordButton.disabled = true;
  try {
    let expired = false;
    const request = navigator.mediaDevices.getUserMedia({ audio: true });
    request.then(mic => { if (expired) mic.getTracks().forEach(track => track.stop()); }).catch(() => {});
    let permissionTimer;
    try {
      stream = await Promise.race([
        request,
        new Promise((_, reject) => {
          permissionTimer = setTimeout(() => {
            expired = true;
            reject(new Error('Microphone permission timed out. Check browser permissions, then try again.'));
          }, 15000);
        }),
      ]);
    } finally { clearTimeout(permissionTimer); }
    if (!window.MediaRecorder) throw new Error('This browser does not support microphone recording. Upload a WAV file instead.');
    const chunks = [];
    const mimeType = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4']
      .find(type => MediaRecorder.isTypeSupported(type));
    mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
    mediaRecorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    mediaRecorder.onstop = async () => {
      clearInterval(ticker);
      stream.getTracks().forEach(track => track.stop());
      recordButton.textContent = '● Start recording';
      recordButton.disabled = false;
      try {
        const source = new Blob(chunks, { type: mediaRecorder.mimeType });
        const Context = window.AudioContext || window.webkitAudioContext;
        if (!Context) throw new Error('Audio decoding is unavailable in this browser.');
        const context = new Context();
        const decoded = await context.decodeAudioData(await source.arrayBuffer());
        const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
        const player = offline.createBufferSource(); player.buffer = decoded;
        player.connect(offline.destination); player.start();
        const rendered = await offline.startRendering();
        const pcm = rendered.getChannelData(0);
        const buffer = new ArrayBuffer(44 + pcm.length * 2);
        const view = new DataView(buffer);
        const word = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
        word(0, 'RIFF'); view.setUint32(4, 36 + pcm.length * 2, true);
        word(8, 'WAVE'); word(12, 'fmt '); view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); view.setUint16(22, 1, true);
        view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
        view.setUint16(32, 2, true); view.setUint16(34, 16, true);
        word(36, 'data'); view.setUint32(40, pcm.length * 2, true);
        pcm.forEach((value, i) => view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, value)) * 32767), true));
        showFile(new File([buffer], 'recording.wav', { type: 'audio/wav' }));
        await context.close();
      } catch (error) { message.textContent = `Could not process recording: ${error.message}`; }
    };
    mediaRecorder.onerror = () => {
      clearInterval(ticker);
      stream.getTracks().forEach(track => track.stop());
      recordButton.textContent = '● Start recording';
      recordButton.disabled = false;
      message.textContent = 'Recording failed. Check microphone permissions or upload an audio file.';
    };
    mediaRecorder.start();
    started = Date.now();
    recordButton.textContent = '■ Stop recording';
    recordButton.disabled = false;
    message.textContent = 'Recording…';
    ticker = setInterval(() => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      timer.textContent = `${seconds}s / 30s`;
      if (seconds >= 30 && mediaRecorder.state === 'recording') mediaRecorder.stop();
    }, 250);
  } catch (error) {
    if (stream) stream.getTracks().forEach(track => track.stop());
    recordButton.disabled = false;
    message.textContent = `Microphone unavailable: ${error.message}`;
  }
});

analyze.addEventListener('click', async () => {
  if (!selectedFile) return;
  analyze.disabled = true;
  message.textContent = 'Analyzing audio…';
  const form = new FormData(); form.append('audio', selectedFile);
  try {
    const response = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Analysis failed.');
    document.querySelector('#result-title').textContent = data.uncertain
      ? 'Uncertain result' : (displayLabels[data.accent] || data.accent);
    document.querySelector('#result-detail').textContent = data.uncertain
      ? 'The model could not make a confident prediction. Try a longer, clearer sample.'
      : `Estimated match: ${Math.round(data.confidence * 100)}%. Based on ${data.speech_seconds}s of detected speech.`;
    const rows = document.querySelector('#alternatives'); rows.replaceChildren();
    data.alternatives.forEach(item => {
      const row = document.createElement('div'); row.className = 'bar-row';
      const label = document.createElement('span');
      const name = document.createElement('b'); name.textContent = displayLabels[item.accent] || item.accent;
      const percent = document.createElement('small'); percent.textContent = `${Math.round(item.confidence * 100)}%`;
      label.append(name, percent);
      const bar = document.createElement('div'); bar.className = 'bar';
      const fill = document.createElement('i'); fill.style.width = `${Math.round(item.confidence * 100)}%`;
      bar.append(fill); row.append(label, bar); rows.append(row);
    });
    result.hidden = false; message.textContent = '';
  } catch (error) { message.textContent = error.message; }
  finally { analyze.disabled = false; }
});

fetch('/api/health').then(r => r.json()).then(data => {
  document.querySelector('#model-status').textContent = data.model_loaded
    ? `Ready · ${data.labels.length} groups` : 'Model needs training';
  if (!data.model_loaded && !recordButton.disabled) message.textContent = 'Train a model to enable analysis. See the README.';
}).catch(() => { document.querySelector('#model-status').textContent = 'Server unavailable'; });
