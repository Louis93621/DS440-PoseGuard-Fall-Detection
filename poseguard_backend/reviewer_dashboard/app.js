const state = {
  snapshot: null,
  latestPose: null,
  latestInference: null,
  latestEvent: null,
  events: [],
  probabilityHistory: [],
  ws: null,
  reconnectTimer: null,
};

const elements = {
  connectionBadge: document.getElementById('connectionBadge'),
  cameraIdPill: document.getElementById('cameraIdPill'),
  frameIdPill: document.getElementById('frameIdPill'),
  pipelineReadyPill: document.getElementById('pipelineReadyPill'),
  skeletonCanvas: document.getElementById('skeletonCanvas'),
  canvasEmpty: document.getElementById('canvasEmpty'),
  poseConfValue: document.getElementById('poseConfValue'),
  poseDetectedValue: document.getElementById('poseDetectedValue'),
  poseFpsValue: document.getElementById('poseFpsValue'),
  videoPathValue: document.getElementById('videoPathValue'),
  eventStateBadge: document.getElementById('eventStateBadge'),
  probabilityValue: document.getElementById('probabilityValue'),
  probabilityBar: document.getElementById('probabilityBar'),
  thresholdMarker: document.getElementById('thresholdMarker'),
  thresholdLabel: document.getElementById('thresholdLabel'),
  windowLabel: document.getElementById('windowLabel'),
  probabilityCanvas: document.getElementById('probabilityCanvas'),
  predictionLabel: document.getElementById('predictionLabel'),
  eventPeakValue: document.getElementById('eventPeakValue'),
  currentEventValue: document.getElementById('currentEventValue'),
  cooldownValue: document.getElementById('cooldownValue'),
  readyValue: document.getElementById('readyValue'),
  inferenceReadyValue: document.getElementById('inferenceReadyValue'),
  eventReadyValue: document.getElementById('eventReadyValue'),
  lastErrorValue: document.getElementById('lastErrorValue'),
  frameQueueValue: document.getElementById('frameQueueValue'),
  poseQueueValue: document.getElementById('poseQueueValue'),
  inferenceInQueueValue: document.getElementById('inferenceInQueueValue'),
  inferenceOutQueueValue: document.getElementById('inferenceOutQueueValue'),
  eventsTableBody: document.getElementById('eventsTableBody'),
  eventsEmpty: document.getElementById('eventsEmpty'),
  refreshEventsBtn: document.getElementById('refreshEventsBtn'),
};

const EDGES = [
  ['nose', 'left_eye'], ['nose', 'right_eye'], ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
  ['left_shoulder', 'right_shoulder'], ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'], ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'], ['left_hip', 'right_hip'], ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'], ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
];

function wsUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/status`;
}

function formatTime(tsMs) {
  if (!tsMs) return '—';
  return new Date(tsMs).toLocaleTimeString();
}

function formatProbability(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(3);
}

function shortEventId(id) {
  if (!id) return '—';
  return id.length > 18 ? `${id.slice(0, 18)}…` : id;
}

function setBadge(el, text, cls) {
  el.textContent = text;
  el.className = cls;
}

function setStateBadge(stateName) {
  const normalized = (stateName || 'NORMAL').toUpperCase();
  const cls = normalized === 'CONFIRMED'
    ? 'state-badge state-confirmed'
    : normalized === 'VERIFYING'
      ? 'state-badge state-verifying'
      : normalized === 'SUSPECTED'
        ? 'state-badge state-suspected'
        : 'state-badge state-normal';
  elements.eventStateBadge.className = cls;
  elements.eventStateBadge.textContent = normalized;
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function drawSkeleton() {
  const { ctx, width, height } = resizeCanvas(elements.skeletonCanvas);
  ctx.clearRect(0, 0, width, height);

  const pose = state.latestPose;
  if (!pose || !Array.isArray(pose.keypoints) || pose.keypoints.length === 0) {
    elements.canvasEmpty.style.display = 'grid';
    return;
  }
  elements.canvasEmpty.style.display = 'none';

  const poseWidth = Number(pose.frame_width) || null;
  const poseHeight = Number(pose.frame_height) || null;
  const points = new Map();
  pose.keypoints.forEach((kp) => points.set(kp.name, kp));

  const pad = 18;
  const effectiveWidth = width - pad * 2;
  const effectiveHeight = height - pad * 2;

  const toCanvas = (kp) => {
    const x = Number(kp.x);
    const y = Number(kp.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    if (poseWidth && poseHeight) {
      return {
        x: pad + (x / poseWidth) * effectiveWidth,
        y: pad + (y / poseHeight) * effectiveHeight,
      };
    }
    return { x, y };
  };

  ctx.lineWidth = 2.5;
  ctx.strokeStyle = '#375d66';
  EDGES.forEach(([a, b]) => {
    const kpA = points.get(a);
    const kpB = points.get(b);
    if (!kpA || !kpB) return;
    if ((kpA.conf ?? 0) < 0.2 || (kpB.conf ?? 0) < 0.2) return;
    const pA = toCanvas(kpA);
    const pB = toCanvas(kpB);
    if (!pA || !pB) return;
    ctx.beginPath();
    ctx.moveTo(pA.x, pA.y);
    ctx.lineTo(pB.x, pB.y);
    ctx.stroke();
  });

  pose.keypoints.forEach((kp) => {
    if ((kp.conf ?? 0) < 0.2) return;
    const p = toCanvas(kp);
    if (!p) return;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4.2, 0, Math.PI * 2);
    ctx.fillStyle = '#8f4a43';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2.2, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
  });
}

function drawProbabilitySparkline() {
  const { ctx, width, height } = resizeCanvas(elements.probabilityCanvas);
  ctx.clearRect(0, 0, width, height);

  const values = state.probabilityHistory;
  if (!values.length) {
    ctx.fillStyle = '#6d665e';
    ctx.font = '14px Inter, sans-serif';
    ctx.fillText('Waiting for inference windows…', 18, height / 2);
    return;
  }

  const padX = 14;
  const padY = 14;
  const chartWidth = width - padX * 2;
  const chartHeight = height - padY * 2;

  ctx.strokeStyle = 'rgba(23,21,19,0.12)';
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach((level) => {
    const y = padY + chartHeight - chartHeight * level;
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(width - padX, y);
    ctx.stroke();
  });

  const threshold = Number(state.snapshot?.latest_inference?.threshold ?? state.latestInference?.threshold ?? 0.5);
  const thresholdY = padY + chartHeight - chartHeight * threshold;
  ctx.strokeStyle = 'rgba(23,21,19,0.55)';
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(padX, thresholdY);
  ctx.lineTo(width - padX, thresholdY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.beginPath();
  values.forEach((value, index) => {
    const x = padX + (index / Math.max(1, values.length - 1)) * chartWidth;
    const y = padY + chartHeight - chartHeight * Math.max(0, Math.min(1, value));
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = '#375d66';
  ctx.stroke();

  const last = values[values.length - 1];
  const x = padX + chartWidth;
  const y = padY + chartHeight - chartHeight * Math.max(0, Math.min(1, last));
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fillStyle = '#8f4a43';
  ctx.fill();
}

function pushProbability(probability) {
  const value = Number(probability);
  if (!Number.isFinite(value)) return;
  const history = state.probabilityHistory;
  const last = history[history.length - 1];
  if (last !== undefined && Math.abs(last - value) < 1e-9) return;
  history.push(value);
  while (history.length > 80) history.shift();
}

function renderStatus() {
  const snapshot = state.snapshot || {};
  const latestPose = state.latestPose;
  const latestInference = state.latestInference;
  const latestEvent = state.latestEvent;
  const eventManager = snapshot.event_manager || {};
  const eventState = eventManager.state || latestEvent?.state || 'NORMAL';
  const threshold = Number(latestInference?.threshold ?? snapshot.latest_inference?.threshold ?? 0.5);
  const probability = Number(latestInference?.fall_probability ?? 0);

  setStateBadge(eventState);
  elements.probabilityValue.textContent = formatProbability(probability);
  elements.probabilityBar.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
  elements.thresholdMarker.style.left = `${Math.max(0, Math.min(100, threshold * 100))}%`;
  elements.thresholdLabel.textContent = `threshold ${threshold.toFixed(2)}`;
  elements.windowLabel.textContent = latestInference ? `window ${latestInference.window_index}` : 'window —';
  elements.predictionLabel.textContent = latestInference?.predicted_label_name || '—';
  elements.eventPeakValue.textContent = latestEvent ? formatProbability(latestEvent.peak_probability) : '—';
  elements.currentEventValue.textContent = latestEvent?.event_id || eventManager.current_event_id || '—';

  const cooldownUntil = Number(eventManager.cooldown_until_ms || 0);
  if (cooldownUntil > Date.now()) {
    const remaining = Math.max(0, Math.ceil((cooldownUntil - Date.now()) / 1000));
    elements.cooldownValue.textContent = `${remaining}s remaining`;
  } else {
    elements.cooldownValue.textContent = 'inactive';
  }

  elements.cameraIdPill.textContent = `camera ${latestPose?.camera_id || snapshot.camera?.camera_id || '—'}`;
  elements.frameIdPill.textContent = `frame ${latestPose?.frame_id ?? '—'}`;
  elements.pipelineReadyPill.textContent = snapshot.ready ? 'pipeline ready' : 'pipeline warming';
  elements.pipelineReadyPill.className = snapshot.ready ? 'chip badge-ok' : 'chip badge-muted';

  elements.poseConfValue.textContent = latestPose ? formatProbability(latestPose.pose_conf) : '—';
  elements.poseDetectedValue.textContent = latestPose?.detected ? 'yes' : latestPose ? 'no' : '—';
  elements.poseFpsValue.textContent = latestPose?.source_fps ? `${Number(latestPose.source_fps).toFixed(1)} fps` : '—';
  elements.videoPathValue.textContent = latestPose?.video_path || snapshot.camera?.video_path || '—';

  elements.readyValue.textContent = snapshot.ready ? 'true' : 'false';
  elements.inferenceReadyValue.textContent = snapshot.inference_ready ? 'true' : 'false';
  elements.eventReadyValue.textContent = snapshot.event_ready ? 'true' : 'false';
  elements.lastErrorValue.textContent = snapshot.last_error || 'none';
  elements.frameQueueValue.textContent = String(snapshot.frame_queue_size ?? '—');
  elements.poseQueueValue.textContent = String(snapshot.pose_queue_size ?? '—');
  elements.inferenceInQueueValue.textContent = String(snapshot.inference_input_queue_size ?? '—');
  elements.inferenceOutQueueValue.textContent = String(snapshot.inference_output_queue_size ?? '—');
}

function renderEvents() {
  const tbody = elements.eventsTableBody;
  tbody.innerHTML = '';
  const items = Array.isArray(state.events) ? state.events : [];
  elements.eventsEmpty.style.display = items.length ? 'none' : 'grid';

  items.forEach((event) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="code-pill" title="${event.event_id}">${shortEventId(event.event_id)}</span></td>
      <td>${event.state || '—'}</td>
      <td>${event.ack_status || '—'}</td>
      <td>${formatProbability(event.peak_probability)}</td>
      <td>${formatTime(event.confirmed_at_ts_ms)}</td>
      <td><div class="row-actions"></div></td>
    `;
    const actions = tr.querySelector('.row-actions');
    const ackButton = document.createElement('button');
    ackButton.className = 'action-btn';
    ackButton.type = 'button';
    ackButton.textContent = event.ack_status === 'ACKNOWLEDGED' ? 'ACKED' : 'ACK';
    ackButton.disabled = event.ack_status === 'ACKNOWLEDGED';
    ackButton.addEventListener('click', async () => {
      ackButton.disabled = true;
      try {
        await fetch(`/api/v1/events/${encodeURIComponent(event.event_id)}/ack`, { method: 'POST' });
        await refreshEvents();
      } catch (err) {
        console.error('ACK failed', err);
      }
    });
    actions.appendChild(ackButton);
    tbody.appendChild(tr);
  });
}

async function refreshStatus() {
  try {
    const res = await fetch('/api/v1/status');
    if (!res.ok) return;
    const payload = await res.json();
    applySnapshot(payload);
  } catch (err) {
    console.error('Status fetch failed', err);
  }
}

async function refreshEvents() {
  try {
    const res = await fetch('/api/v1/events?limit=50');
    if (!res.ok) return;
    const payload = await res.json();
    state.events = payload.items || [];
    renderEvents();
  } catch (err) {
    console.error('Events fetch failed', err);
  }
}

function applySnapshot(snapshot) {
  state.snapshot = snapshot;
  if (snapshot.latest_pose) state.latestPose = snapshot.latest_pose;
  if (snapshot.latest_inference) {
    state.latestInference = snapshot.latest_inference;
    pushProbability(snapshot.latest_inference.fall_probability);
  }
  if (snapshot.latest_event) state.latestEvent = snapshot.latest_event;
  renderAll();
}

function handleSocketMessage(message) {
  const payload = typeof message === 'string' ? JSON.parse(message) : message;
  switch (payload.type) {
    case 'hello':
    case 'heartbeat':
      applySnapshot(payload.data || {});
      break;
    case 'pose_frame':
      state.latestPose = payload.data;
      renderAll();
      break;
    case 'inference_result':
      state.latestInference = payload.data;
      pushProbability(payload.data?.fall_probability);
      renderAll();
      break;
    case 'fall_event':
      if (payload.data?.event) state.latestEvent = payload.data.event;
      renderAll();
      refreshEvents();
      break;
    case 'error':
      console.error('Backend error', payload.data);
      break;
    default:
      break;
  }
}

function connectWebSocket() {
  if (state.ws) {
    try { state.ws.close(); } catch (_) {}
  }
  const socket = new WebSocket(wsUrl());
  state.ws = socket;
  setBadge(elements.connectionBadge, 'Connecting', 'badge badge-muted');

  socket.addEventListener('open', () => {
    setBadge(elements.connectionBadge, 'Live', 'badge badge-live');
  });
  socket.addEventListener('message', (event) => {
    handleSocketMessage(event.data);
  });
  socket.addEventListener('close', () => {
    setBadge(elements.connectionBadge, 'Reconnecting', 'badge badge-muted');
    if (state.reconnectTimer) clearTimeout(state.reconnectTimer);
    state.reconnectTimer = setTimeout(connectWebSocket, 1500);
  });
  socket.addEventListener('error', () => {
    setBadge(elements.connectionBadge, 'Connection Error', 'badge badge-error');
    socket.close();
  });
}

function renderAll() {
  renderStatus();
  drawSkeleton();
  drawProbabilitySparkline();
}

function init() {
  elements.refreshEventsBtn.addEventListener('click', refreshEvents);
  window.addEventListener('resize', renderAll);
  refreshStatus();
  refreshEvents();
  connectWebSocket();
  setInterval(refreshStatus, 5000);
  setInterval(refreshEvents, 10000);
}

init();
