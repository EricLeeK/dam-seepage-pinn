import React, { useRef, useEffect, useCallback } from 'react';

const CANVAS_W = 600;
const CANVAS_H = 500;
const PADDING = 40;
const SNAP = 0.5;
const LABELS = ['左下', '右下', '左上', '右上'];

const C = {
  grid: '#e8e8e8',
  gridBold: '#c8c8c8',
  damFill: '#e8e8e8',
  damFillHover: '#dde5f0',
  damBorder: '#2d3748',
  vertex: '#e53e3e',
  vertexHover: '#fc8181',
  water: 'rgba(66, 153, 225, 0.25)',
  waterLine: '#3182ce',
  waterText: '#2b6cb0',
  axis: '#718096',
  axisText: '#4a5568',
  labelBg: 'rgba(255,255,255,0.85)',
};

function snap(v) { return Math.round(v / SNAP) * SNAP; }

// Build trapezoid vertices from params: [BL, BR, TL, TR]
// Supports b_bottom < b_top (inverted trapezoid)
function buildVertices(p) {
  const { h, b_top, b_bottom, angle_up, angle_down } = p;
  const ru = (angle_up * Math.PI) / 180;
  const rd = (angle_down * Math.PI) / 180;
  const xou = h / Math.tan(ru);
  const xod = h / Math.tan(rd);
  return [
    [0, 0],               // BL
    [b_bottom, 0],        // BR
    [xou, h],             // TL
    [xou + b_top, h],     // TR
  ];
}

// Extract params from vertices
function extractParams(verts) {
  const [v0, v1, v2, v3] = verts;
  const h = Math.max(v2[1], v3[1], 1);
  const b_bottom = Math.abs(v1[0] - v0[0]);
  const b_top = Math.abs(v3[0] - v2[0]);
  // Angles from horizontal
  const leftDx = v2[0] - v0[0];
  const rightDx = v1[0] - v3[0];
  const au = Math.atan2(h, Math.abs(leftDx)) * 180 / Math.PI;
  const ad = Math.atan2(h, Math.abs(rightDx)) * 180 / Math.PI;
  return {
    h: Math.round(h * 100) / 100,
    b_top: Math.round(b_top * 100) / 100,
    b_bottom: Math.round(b_bottom * 100) / 100,
    angle_up: Math.round(au * 100) / 100,
    angle_down: Math.round(ad * 100) / 100,
  };
}

function CanvasEditor({ params, onParamsChange }) {
  const canvasRef = useRef(null);
  const stateRef = useRef({
    verts: [],           // [BL, BR, TL, TR] in LOGICAL coords
    scale: 1,
    ox: PADDING,
    oy: CANVAS_H - PADDING,
    dragging: null,      // index into verts (0-3)
    hover: null,         // index into verts (0-3)
  });
  const isDraggingRef = useRef(false);
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const toCanvas = useCallback((x, y, s, ox, oy) => [ox + x * s, oy - y * s], []);
  const toLogical = useCallback((px, py, s, ox, oy) => [(px - ox) / s, (oy - py) / s], []);

  // ── Draw ──
  const draw = useCallback(() => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext('2d');
    const st = stateRef.current;
    const { verts, scale, ox, oy } = st;
    const p = paramsRef.current;

    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    const maxXM = Math.ceil((CANVAS_W - 2 * PADDING) / scale);
    const maxYM = Math.ceil((CANVAS_H - 2 * PADDING) / scale);

    // ── Grid ──
    ctx.strokeStyle = C.grid; ctx.lineWidth = 0.5;
    for (let xm = 0; xm <= maxXM; xm++) {
      const [px] = toCanvas(xm, 0, scale, ox, oy);
      ctx.beginPath(); ctx.moveTo(px, oy); ctx.lineTo(px, oy - maxYM * scale); ctx.stroke();
    }
    for (let ym = 0; ym <= maxYM; ym++) {
      const [, py] = toCanvas(0, ym, scale, ox, oy);
      ctx.beginPath(); ctx.moveTo(ox, py); ctx.lineTo(ox + maxXM * scale, py); ctx.stroke();
    }
    ctx.strokeStyle = C.gridBold; ctx.lineWidth = 1;
    for (let xm = 0; xm <= maxXM; xm += 5) {
      const [px] = toCanvas(xm, 0, scale, ox, oy);
      ctx.beginPath(); ctx.moveTo(px, oy); ctx.lineTo(px, oy - maxYM * scale); ctx.stroke();
    }
    for (let ym = 0; ym <= maxYM; ym += 5) {
      const [, py] = toCanvas(0, ym, scale, ox, oy);
      ctx.beginPath(); ctx.moveTo(ox, py); ctx.lineTo(ox + maxXM * scale, py); ctx.stroke();
    }

    // ── Axes ──
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + maxXM * scale, oy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox, oy - maxYM * scale); ctx.stroke();
    ctx.fillStyle = C.axisText; ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    for (let xm = 0; xm <= maxXM; xm += 5) {
      const [px] = toCanvas(xm, 0, scale, ox, oy);
      ctx.fillText(`${xm}m`, px, oy + 16);
    }
    ctx.textAlign = 'right';
    for (let ym = 0; ym <= maxYM; ym += 5) {
      const [, py] = toCanvas(0, ym, scale, ox, oy);
      ctx.fillText(`${ym}m`, ox - 6, py + 4);
    }
    ctx.fillText('高程(m)', ox - 10, oy - maxYM * scale + 14);
    ctx.textAlign = 'center';
    ctx.fillText('宽度(m)', ox + maxXM * scale / 2, oy + 32);

    if (!verts.length) return;

    const [v0, v1, v2, v3] = verts;

    // ── Water upstream ──
    // Left slope: v0(BL) → v2(TL). Intersection at h_up:
    //   xL = v0[0] + (h_up / v2[1]) * (v2[0] - v0[0])
    // Water fills from left grid edge (0) to xL
    if (p.h_up > 0 && v2[1] > 0) {
      const clampedH = Math.min(p.h_up, v2[1]);
      const xL = v0[0] + (clampedH / v2[1]) * (v2[0] - v0[0]);
      const gridLeft = 0;
      const pts = [
        toCanvas(gridLeft, p.h_up, scale, ox, oy),
        toCanvas(xL, p.h_up, scale, ox, oy),
        toCanvas(xL, 0, scale, ox, oy),
        toCanvas(gridLeft, 0, scale, ox, oy),
      ];
      ctx.fillStyle = C.water;
      ctx.beginPath(); ctx.moveTo(...pts[0]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(...pts[i]);
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = C.waterLine; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(...pts[0]); ctx.lineTo(...pts[1]); ctx.stroke();
      ctx.fillStyle = C.waterText; ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${p.h_up}m`, pts[0][0] + 40, pts[0][1] - 4);
    }

    // ── Water downstream ──
    // Right slope: v3(TR) → v1(BR). Intersection at h_down:
    //   xR = v1[0] - (h_down / v3[1]) * (v1[0] - v3[0])
    // Water fills from xR to right grid edge (maxXM)
    if (p.h_down > 0 && v3[1] > 0) {
      const clampedH = Math.min(p.h_down, v3[1]);
      const xR = v1[0] - (clampedH / v3[1]) * (v1[0] - v3[0]);
      const gridRight = maxXM;
      const pts = [
        toCanvas(xR, p.h_down, scale, ox, oy),
        toCanvas(gridRight, p.h_down, scale, ox, oy),
        toCanvas(gridRight, 0, scale, ox, oy),
        toCanvas(xR, 0, scale, ox, oy),
      ];
      ctx.fillStyle = C.water;
      ctx.beginPath(); ctx.moveTo(...pts[0]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(...pts[i]);
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = C.waterLine; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(...pts[0]); ctx.lineTo(...pts[1]); ctx.stroke();
      ctx.fillStyle = C.waterText; ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`${p.h_down}m`, pts[1][0] - 40, pts[1][1] - 4);
      ctx.textAlign = 'center';
    }

    // ── Trapezoid fill (perimeter: BL→BR→TR→TL) ──
    const allCV = verts.map((v) => toCanvas(v[0], v[1], scale, ox, oy));
    const perimeter = [allCV[0], allCV[1], allCV[3], allCV[2]]; // BL, BR, TR, TL
    ctx.fillStyle = C.damFill;
    ctx.beginPath(); ctx.moveTo(...perimeter[0]);
    for (let i = 1; i < perimeter.length; i++) ctx.lineTo(...perimeter[i]);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = C.damBorder; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(...perimeter[0]);
    for (let i = 1; i < perimeter.length; i++) ctx.lineTo(...perimeter[i]);
    ctx.closePath(); ctx.stroke();

    // ── Vertices (draw in original order: 0=BL,1=BR,2=TL,3=TR) ──
    const labelOffsets = [
      [-2, 16],   // 左下: label below-left
      [-2, 16],   // 右下: label below-right
      [-2, -12],  // 左上: label above
      [-2, -12],  // 右上: label above
    ];
    allCV.forEach(([px, py], i) => {
      const isHover = st.hover === i;
      const r = isHover ? 8 : 6;
      ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = isHover ? C.vertexHover : C.vertex;
      ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = '#555'; ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      const [ox2, oy2] = labelOffsets[i];
      ctx.fillText(LABELS[i], px + ox2, py + oy2);
    });
  }, [toCanvas]);

  // ── Rebuild vertices when params change from outside ──
  useEffect(() => {
    if (isDraggingRef.current) return;
    const st = stateRef.current;
    const newVerts = buildVertices(params);
    st.verts = newVerts;
    const maxDim = Math.max(params.b_bottom + 16, params.h + 8);
    st.scale = Math.min((CANVAS_W - 2 * PADDING) / maxDim, (CANVAS_H - 2 * PADDING) / maxDim, 14);
    const cx = (newVerts[0][0] + newVerts[1][0]) / 2;
    st.ox = CANVAS_W / 2 - cx * st.scale;
    st.oy = CANVAS_H - PADDING - 10;
    draw();
  }, [params.h, params.b_top, params.b_bottom, params.angle_up, params.angle_down, draw]);

  // ── Mouse handlers ──
  const getPos = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  // findVertex returns VERTEX INDEX (0=BL,1=BR,2=TL,3=TR) — NOT draw order
  const findVertex = (px, py) => {
    const st = stateRef.current;
    const allCV = st.verts.map((v) => toCanvas(v[0], v[1], st.scale, st.ox, st.oy));
    let best = -1, bestD = 15;
    allCV.forEach(([vx, vy], i) => {
      const d = Math.hypot(px - vx, py - vy);
      if (d < bestD) { bestD = d; best = i; }
    });
    return best;
  };

  const handleMove = (e) => {
    const st = stateRef.current;
    const [px, py] = getPos(e);

    if (st.dragging !== null) {
      const idx = st.dragging;
      const [lx, ly] = toLogical(px, py, st.scale, st.ox, st.oy);
      const nv = st.verts.map((v) => [...v]);

      if (idx === 0) {
        // BL: free x, y=0
        nv[0][0] = Math.max(0, snap(lx));
        nv[0][1] = 0;
      } else if (idx === 1) {
        // BR: free x, y=0
        nv[1][0] = snap(lx);
        nv[1][1] = 0;
      } else if (idx === 2) {
        // TL: free x,y
        nv[2][0] = snap(lx);
        nv[2][1] = Math.max(1, snap(ly));
        nv[3][1] = nv[2][1]; // TR follows TL's y (horizontal top)
      } else if (idx === 3) {
        // TR: free x,y
        nv[3][0] = snap(lx);
        nv[3][1] = Math.max(1, snap(ly));
        nv[2][1] = nv[3][1]; // TL follows TR's y (horizontal top)
      }

      st.verts = nv;
      const newParams = extractParams(nv);
      paramsRef.current = { ...paramsRef.current, ...newParams };
      onParamsChange(newParams);
      draw();
      return;
    }

    // Hover
    const nearest = findVertex(px, py);
    const prev = st.hover;
    st.hover = nearest >= 0 ? nearest : null;
    if (prev !== st.hover) {
      canvasRef.current.style.cursor = nearest >= 0 ? 'grab' : 'default';
      draw();
    }
  };

  const handleDown = (e) => {
    const st = stateRef.current;
    const [px, py] = getPos(e);
    const nearest = findVertex(px, py);
    if (nearest >= 0) {
      st.dragging = nearest;
      isDraggingRef.current = true;
      canvasRef.current.style.cursor = 'grabbing';
    }
  };

  const handleUp = () => {
    stateRef.current.dragging = null;
    isDraggingRef.current = false;
    if (canvasRef.current) {
      canvasRef.current.style.cursor = stateRef.current.hover !== null ? 'grab' : 'default';
    }
  };

  const handleLeave = () => {
    const st = stateRef.current;
    st.dragging = null; st.hover = null;
    isDraggingRef.current = false;
    draw();
  };

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        style={{
          border: '1px solid #d0d0d0',
          borderRadius: 10,
          background: '#fafbfc',
          cursor: 'default',
          boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        }}
        onMouseMove={handleMove}
        onMouseDown={handleDown}
        onMouseUp={handleUp}
        onMouseLeave={handleLeave}
      />
      <div style={{
        position: 'absolute', bottom: 10, left: 10,
        fontSize: 11, color: '#999',
        background: C.labelBg, padding: '3px 8px',
        borderRadius: 6, pointerEvents: 'none',
      }}>
        🖱 拖拽顶点调整坝体 · 吸附 0.5m 网格
      </div>
    </div>
  );
}

export default CanvasEditor;
