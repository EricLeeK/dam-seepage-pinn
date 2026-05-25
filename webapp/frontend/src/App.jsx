import React, { useState, useCallback, useRef, useEffect } from 'react';
import CanvasEditor from './components/CanvasEditor.jsx';

// ── Default geometry ──
const DEFAULT_GEOMETRY = {
  h: 30,
  b_top: 10,
  b_bottom: 40,
  angle_up: 60,
  angle_down: 45,
};

const DEFAULT_HYDRAULIC = {
  h_up: 25,
  h_down: 5,
  permeability_k: 1.0,
};

const DEFAULT_TRAINING = {
  outer_iters: 30,
  num_domain: 10000,
  num_boundary: 2000,
  adam_epochs: 1000,
  lbfgs_max_iter: 5000,
};

const PRESET_FAST = {
  outer_iters: 3,
  num_domain: 2000,
  num_boundary: 500,
  adam_epochs: 200,
  lbfgs_max_iter: 200,
};

const PRESET_DEFAULT = { ...DEFAULT_TRAINING };

// ── API helpers ──
const API_BASE = '/api';

async function apiPost(path, body) {
  const res = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(JSON.stringify(err));
  }
  return res.json();
}

async function apiUpload(path, formData) {
  const res = await fetch(API_BASE + path, { method: 'POST', body: formData });
  return res.json();
}

async function apiGet(path) {
  const res = await fetch(API_BASE + path);
  return res.json();
}

// ── Components ──

function InputGroup({ label, value, onChange, type = 'number', min, max, step = 0.1, unit = '' }) {
  return (
    <div style={styles.inputGroup}>
      <label style={styles.label}>{label}</label>
      <div style={styles.inputWrap}>
        <input
          type={type}
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => onChange(type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value)}
          style={styles.input}
        />
        {unit && <span style={styles.unit}>{unit}</span>}
      </div>
    </div>
  );
}

function Section({ title, children, style = {}, onClick }) {
  return (
    <div style={{ ...styles.section, ...style }} onClick={onClick}>
      <h3 style={styles.sectionTitle}>{title}</h3>
      {children}
    </div>
  );
}

function App() {
  // ── State ──
  const [mode, setMode] = useState('draw'); // 'draw' | 'sketch'
  const [geometry, setGeometry] = useState({ ...DEFAULT_GEOMETRY });
  const [hydraulic, setHydraulic] = useState({ ...DEFAULT_HYDRAULIC });
  const [training, setTraining] = useState({ ...DEFAULT_TRAINING });
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Sketch mode
  const [sketchFile, setSketchFile] = useState(null);
  const [sketchPreview, setSketchPreview] = useState(null);
  const [agentResult, setAgentResult] = useState(null);
  const [agentLoading, setAgentLoading] = useState(false);

  // Solver
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);
  const [solverLoading, setSolverLoading] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [resultImage, setResultImage] = useState(null);
  const [resultLoss, setResultLoss] = useState(null);
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const pollingRef = useRef(null);

  // ── Geometry param change from input fields ──
  const updateGeo = useCallback((key, val) => {
    setGeometry((g) => {
      const next = { ...g, [key]: val };
      // Auto-derive bottom width from angles if they changed
      if (key === 'h' || key === 'b_top' || key === 'angle_up' || key === 'angle_down') {
        const ru = (next.angle_up * Math.PI) / 180;
        const rd = (next.angle_down * Math.PI) / 180;
        const xou = next.h / Math.tan(ru);
        const xod = next.h / Math.tan(rd);
        next.b_bottom = Math.round((xou + next.b_top + xod) * 100) / 100;
      }
      return next;
    });
  }, []);

  // Geometry change from Canvas (dragging vertices)
  const handleCanvasParamsChange = useCallback((newParams) => {
    setGeometry((g) => ({ ...g, ...newParams }));
  }, []);

  // ── Agent Analysis ──
  const runAgent = async () => {
    if (!sketchFile) return alert('请先上传草图');
    setAgentLoading(true);
    setAgentResult(null);
    const fd = new FormData();
    fd.append('file', sketchFile);
    // API key is read from .env on the backend
    try {
      const data = await apiUpload('/agent/analyze', fd);
      setAgentResult(data);
      if (data.status === 'success' && data.geometry) {
        const g = { ...data.geometry };
        // 强制从角度和顶宽重算坝底宽度，避免 null 或错误值
        const ru = (g.angle_up * Math.PI) / 180;
        const rd = (g.angle_down * Math.PI) / 180;
        g.b_bottom = Math.round((g.h / Math.tan(ru) + g.b_top + g.h / Math.tan(rd)) * 100) / 100;
        setGeometry(g);
        if (data.hydraulic) {
          setHydraulic({
            h_up: data.hydraulic.upstream_head ?? data.hydraulic.h_up ?? 25,
            h_down: data.hydraulic.downstream_head ?? data.hydraulic.h_down ?? 5,
            permeability_k: data.hydraulic.permeability_k ?? 1.0,
          });
        }
      }
    } catch (e) {
      alert('Agent 分析失败: ' + e.message);
    } finally {
      setAgentLoading(false);
    }
  };

  // ── PINN Solver ──
  const runSolver = async () => {
    setSolverLoading(true);
    setCancelling(false);
    setTaskStatus({ status: 'running', progress: [] });
    setResultImage(null);
    setResultLoss(null);

    const req = {
      geometry: {
        shape: 'trapezoid',
        vertices: [
          [0, 0],
          [geometry.b_bottom, 0],
          [geometry.h / Math.tan((geometry.angle_up * Math.PI) / 180), geometry.h],
          [geometry.h / Math.tan((geometry.angle_up * Math.PI) / 180) + geometry.b_top, geometry.h],
        ],
        h: geometry.h,
        b_top: geometry.b_top,
        b_bottom: geometry.b_bottom,
        angle_up: geometry.angle_up,
        angle_down: geometry.angle_down,
      },
      hydraulic: {
        upstream_head: hydraulic.h_up,
        downstream_head: hydraulic.h_down,
        permeability_k: hydraulic.permeability_k,
      },
      training,
    };

    try {
      const { task_id } = await apiPost('/pinn/solve', req);
      setTaskId(task_id);
      startPolling(task_id);
    } catch (e) {
      alert('启动求解失败: ' + e.message);
      setSolverLoading(false);
    }
  };

  const cancelSolver = async () => {
    if (!taskId) return;
    setCancelling(true);
    try {
      await fetch(`/api/pinn/cancel/${taskId}`, { method: 'POST' });
    } catch (e) {
      console.error('Cancel failed:', e);
      setCancelling(false);
    }
  };

  const startPolling = (tid) => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    pollingRef.current = setInterval(async () => {
      try {
        const data = await apiGet(`/pinn/status/${tid}`);
        setTaskStatus(data);
        if (data.status === 'completed') {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
          setSolverLoading(false);
          setResultImage(`/api/pinn/result/${tid}/plot?t=${Date.now()}`);
          setResultLoss(data.result?.final_loss);
        } else if (data.status === 'failed') {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
          setSolverLoading(false);
          alert('求解失败: ' + (data.error || '未知错误'));
        } else if (data.status === 'cancelled') {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
          setSolverLoading(false);
          setCancelling(false);
        }
      } catch (e) {
        console.error('Polling error:', e);
      }
    }, 1000);
  };

  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  // ── Render ──
  return (
    <div style={styles.app}>
      <style>{`.result-image-wrap:hover .result-overlay{opacity:1}`}</style>
      {/* Header */}
      <header style={styles.header}>
        <h1 style={styles.headerTitle}>🌊 大坝渗流 PINN 分析系统</h1>
        <p style={styles.headerSubtitle}>基于物理信息神经网络的大坝渗流场自动求解</p>
      </header>

      <div style={styles.main}>
        {/* ════════════════ LEFT PANEL (sticky) ════════════════ */}
        <aside style={styles.leftPanel}>
          {/* Mode Switch */}
          <Section title="输入模式">
            <div style={styles.modeSwitch}>
              <button
                style={{ ...styles.modeBtn, ...(mode === 'draw' ? styles.modeBtnActive : {}) }}
                onClick={() => setMode('draw')}
              >
                ✏️ 交互绘制
              </button>
              <button
                style={{ ...styles.modeBtn, ...(mode === 'sketch' ? styles.modeBtnActive : {}) }}
                onClick={() => setMode('sketch')}
              >
                🖼️ 草图识别
              </button>
            </div>
          </Section>

          {/* Sketch Upload */}
          {mode === 'sketch' && (
            <Section title="上传草图">
              <div style={styles.uploadArea}>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files[0];
                    setSketchFile(file);
                    if (sketchPreview) URL.revokeObjectURL(sketchPreview);
                    setSketchPreview(file ? URL.createObjectURL(file) : null);
                  }}
                  style={{ display: 'none' }}
                  id="sketch-upload"
                />
                {sketchPreview ? (
                  <label htmlFor="sketch-upload" style={{ cursor: 'pointer', display: 'block' }}>
                    <img src={sketchPreview} alt="草图预览" style={{ width: '100%', borderRadius: 6, maxHeight: 180, objectFit: 'contain', display: 'block' }} />
                    <div style={{ fontSize: 11, color: '#718096', marginTop: 6, textAlign: 'center' }}>📎 {sketchFile.name} · 点击更换</div>
                  </label>
                ) : (
                  <label htmlFor="sketch-upload" style={styles.uploadLabel}>
                    📁 点击上传大坝草图
                  </label>
                )}
              </div>
              <button
                style={{ ...styles.btn, ...styles.btnPrimary, width: '100%', marginTop: 10 }}
                onClick={runAgent}
                disabled={agentLoading || !sketchFile}
              >
                {agentLoading ? '🔍 Agent 分析中...' : '🚀 运行智能体分析'}
              </button>

              {agentResult && (
                <div style={styles.agentResult}>
                  <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#333' }}>🔍 智能体审计报告</h4>
                  <div style={styles.agentBadge(agentResult.status)}>
                    {agentResult.status === 'success' ? '✅ 验证通过' : '❌ 验证失败'}
                  </div>
                  <p style={{ fontSize: 11, color: '#666', marginTop: 6 }}>
                    {agentResult.validation_report}
                  </p>
                  {agentResult.agent1_data && (
                    <details style={{ marginTop: 8 }}>
                      <summary style={{ fontSize: 11, cursor: 'pointer', color: '#555' }}>
                        查看 Agent 1 原始提取
                      </summary>
                      <pre style={styles.codeBlock}>
                        {JSON.stringify(agentResult.agent1_data, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              )}
            </Section>
          )}

          {/* Geometry Params */}
          <Section title="几何参数">
            <InputGroup label="坝体高度" value={geometry.h} onChange={(v) => updateGeo('h', v)} unit="m" />
            <InputGroup label="坝顶宽度" value={geometry.b_top} onChange={(v) => updateGeo('b_top', v)} unit="m" />
            <InputGroup
              label="坝底宽度"
              value={geometry.b_bottom}
              onChange={(v) => updateGeo('b_bottom', v)}
              unit="m"
              style={{ opacity: 0.7 }}
            />
            <div style={{ fontSize: 10, color: '#999', margin: '-6px 0 6px' }}>
              * 底宽由高度、顶宽和坡角自动计算，也可手动覆盖
            </div>
            <InputGroup label="上游坡角" value={geometry.angle_up} onChange={(v) => updateGeo('angle_up', v)} unit="°" />
            <InputGroup label="下游坡角" value={geometry.angle_down} onChange={(v) => updateGeo('angle_down', v)} unit="°" />
          </Section>

          {/* Hydraulic Params */}
          <Section title="水力参数">
            <InputGroup label="上游总水头" value={hydraulic.h_up} onChange={(v) => setHydraulic((h) => ({ ...h, h_up: v }))} unit="m" />
            <InputGroup label="下游总水头" value={hydraulic.h_down} onChange={(v) => setHydraulic((h) => ({ ...h, h_down: v }))} unit="m" />
          </Section>

          {/* Advanced Settings */}
          <Section title="⚙️ 高级设置" style={{ cursor: 'pointer' }} onClick={() => setAdvancedOpen(!advancedOpen)}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span>展开高级参数</span>
              <span style={{ transform: advancedOpen ? 'rotate(180deg)' : 'rotate(0)', transition: '0.2s' }}>▼</span>
            </div>
          </Section>
          {advancedOpen && (
            <div style={{ padding: '0 16px 12px', background: '#f8f9fa' }}>
              <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
                <button
                  style={{ ...styles.btn, ...styles.btnSecondary, flex: 1, fontSize: 11 }}
                  onClick={() => setTraining({ ...PRESET_FAST })}
                >⚡ 快速测试 (~2min)</button>
                <button
                  style={{ ...styles.btn, ...styles.btnSecondary, flex: 1, fontSize: 11 }}
                  onClick={() => setTraining({ ...PRESET_DEFAULT })}
                >🎯 完整精度 (~30min)</button>
              </div>
              <InputGroup label="渗透系数 K" value={hydraulic.permeability_k} onChange={(v) => setHydraulic((h) => ({ ...h, permeability_k: v }))} step={0.1} />
              <InputGroup label="外层迭代轮数" value={training.outer_iters} onChange={(v) => setTraining((t) => ({ ...t, outer_iters: v }))} step={1} />
              <InputGroup label="域内采样点数" value={training.num_domain} onChange={(v) => setTraining((t) => ({ ...t, num_domain: v }))} step={1000} />
              <InputGroup label="边界采样点数" value={training.num_boundary} onChange={(v) => setTraining((t) => ({ ...t, num_boundary: v }))} step={500} />
              <InputGroup label="Adam 迭代次数" value={training.adam_epochs} onChange={(v) => setTraining((t) => ({ ...t, adam_epochs: v }))} step={100} />
              <InputGroup label="L-BFGS 最大迭代" value={training.lbfgs_max_iter} onChange={(v) => setTraining((t) => ({ ...t, lbfgs_max_iter: v }))} step={500} />
            </div>
          )}

          {/* Run / Cancel Button */}
          <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
            <button
              style={{ ...styles.btn, ...styles.btnPrimary, ...styles.btnLarge, flex: 1 }}
              onClick={runSolver}
              disabled={solverLoading}
            >
              {solverLoading ? '⏳ 求解中...' : '▶️ 启动 PINN 求解'}
            </button>
            {solverLoading && (
              <button
                style={{
                  ...styles.btn, ...styles.btnDanger, ...styles.btnLarge,
                  opacity: cancelling ? 0.6 : 1,
                  cursor: cancelling ? 'not-allowed' : 'pointer',
                }}
                onClick={cancelSolver}
                disabled={cancelling}
              >
                {cancelling ? '⏳ 取消中...' : '⏹ 停止'}
              </button>
            )}
          </div>
        </aside>

        {/* ════════════════ RIGHT SCROLLABLE CONTENT ════════════════ */}
        <div style={styles.rightContent}>
          {/* Canvas Editor + Log Card */}
          <div style={styles.card}>
            <div style={styles.canvasHeader}>
              <h2 style={styles.canvasTitle}>🎨 几何编辑器</h2>
              <span style={styles.canvasHint}>拖拽红色顶点调整坝体形状</span>
            </div>
            <div style={styles.canvasRow}>
              <CanvasEditor
                params={{ ...geometry, h_up: hydraulic.h_up, h_down: hydraulic.h_down }}
                onParamsChange={handleCanvasParamsChange}
              />
              {/* Training Log — inside editor card, right side */}
              {taskStatus && (taskStatus.progress || []).length > 0 && (
                <div style={styles.logPanel}>
                  <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#e2e8f0' }}>📡 训练日志</h4>
                  <div style={styles.logContainer}>
                    {(taskStatus.progress || []).map((p, i) => (
                      <div key={i} style={styles.logLine}>
                        <span style={styles.logTime}>
                          {p.timestamp ? new Date(p.timestamp).toLocaleTimeString('zh-CN', { hour12: false }) : ''}
                        </span>
                        <span style={styles.logMsg}>{p.message || `${p.phase} ${p.current}/${p.total}`}</span>
                      </div>
                    ))}
                    {taskStatus.status === 'running' && (
                      <div style={{ ...styles.logLine, opacity: 0.5 }}>
                        <span style={styles.logTime}>now</span>
                        <span style={styles.logMsg}>⏳ 等待下一步...</span>
                      </div>
                    )}
                    {taskStatus.status === 'completed' && (
                      <div style={{ ...styles.logLine, color: '#48bb78' }}>
                        <span style={styles.logTime}>done</span>
                        <span style={styles.logMsg}>✅ 训练完成！</span>
                      </div>
                    )}
                    {taskStatus.status === 'cancelled' && (
                      <div style={{ ...styles.logLine, color: '#fc8181' }}>
                        <span style={styles.logTime}>done</span>
                        <span style={styles.logMsg}>🛑 已取消</span>
                      </div>
                    )}
                    {taskStatus.status === 'failed' && (
                      <div style={{ ...styles.logLine, color: '#fc8181' }}>
                        <span style={styles.logTime}>error</span>
                        <span style={styles.logMsg}>❌ {taskStatus.error || '训练失败'}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
            <div style={styles.geometryInfo}>
              <div style={styles.infoRow}>
                <span style={styles.infoLabel}>左下</span>
                <span style={styles.infoValue}>(0, 0)</span>
              </div>
              <div style={styles.infoRow}>
                <span style={styles.infoLabel}>右下</span>
                <span style={styles.infoValue}>({geometry.b_bottom}m, 0)</span>
              </div>
              <div style={styles.infoRow}>
                <span style={styles.infoLabel}>左上</span>
                <span style={styles.infoValue}>
                  ({(geometry.h / Math.tan(geometry.angle_up * Math.PI / 180)).toFixed(2)}m, {geometry.h}m)
                </span>
              </div>
              <div style={styles.infoRow}>
                <span style={styles.infoLabel}>右上</span>
                <span style={styles.infoValue}>
                  ({(geometry.h / Math.tan(geometry.angle_up * Math.PI / 180) + geometry.b_top).toFixed(2)}m, {geometry.h}m)
                </span>
              </div>
            </div>
          </div>

          {/* Results Card */}
          <div style={styles.card}>
            <Section title="分析结果">
              {!resultImage && !solverLoading && (
                <div style={styles.emptyResult}>
                  <div style={{ fontSize: 48, marginBottom: 12 }}>📊</div>
                  <p style={{ color: '#888', fontSize: 14 }}>运行求解后将在此显示渗流场分析结果</p>
                </div>
              )}

              {solverLoading && !resultImage && (
                <div style={styles.emptyResult}>
                  <div style={{ fontSize: 48, marginBottom: 12, animation: 'spin 1s linear infinite' }}>⚙️</div>
                  <p style={{ color: '#666', fontSize: 14 }}>PINN 正在训练中，请稍候...</p>
                  <p style={{ color: '#999', fontSize: 11, marginTop: 8 }}>
                    外部空间迭代 + L-BFGS 收敛，通常需要 2-5 分钟
                  </p>
                </div>
              )}

              {resultImage && (
                <>
                  <div
                    className="result-image-wrap"
                    style={styles.resultImageWrap}
                    onClick={() => setLightboxOpen(true)}
                  >
                    <img
                      src={resultImage}
                      alt="PINN Seepage Analysis"
                      style={styles.resultImage}
                    />
                    <div className="result-overlay" style={styles.resultOverlay}>🔍 点击查看大图</div>
                  </div>

                  {resultLoss !== null && (
                    <div style={styles.lossBox}>
                      <span style={styles.lossLabel}>最终物理误差 (MSE):</span>
                      <span style={styles.lossValue}>{resultLoss.toExponential(4)}</span>
                    </div>
                  )}

                  <div style={styles.downloadRow}>
                    <a
                      href={resultImage}
                      download="pinn_seepage_result.png"
                      style={{ ...styles.btn, ...styles.btnSecondary, flex: 1, textDecoration: 'none', textAlign: 'center' }}
                    >
                      📷 下载图片
                    </a>
                    {taskId && (
                      <a
                        href={`/api/pinn/result/${taskId}/npz`}
                        download="seepage_plot_data.npz"
                        style={{ ...styles.btn, ...styles.btnSecondary, flex: 1, textDecoration: 'none', textAlign: 'center' }}
                      >
                        📦 下载数据
                      </a>
                    )}
                  </div>
                </>
              )}
            </Section>
          </div>
        </div>
      </div>

      {/* Lightbox Overlay */}
      {lightboxOpen && (
        <div style={styles.lightbox} onClick={() => setLightboxOpen(false)}>
          <img src={resultImage} alt="PINN Seepage Analysis (Full)" style={styles.lightboxImage} />
        </div>
      )}
    </div>
  );
}

// ── Styles ──
const styles = {
  app: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    overflow: 'hidden',
    background: '#f0f2f5',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
  },
  header: {
    background: '#1a365d',
    color: '#fff',
    padding: '14px 24px',
    flexShrink: 0,
    boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
  },
  headerTitle: {
    margin: 0,
    fontSize: 20,
    fontWeight: 600,
    letterSpacing: '0.5px',
  },
  headerSubtitle: {
    margin: '4px 0 0',
    fontSize: 12,
    opacity: 0.75,
    fontWeight: 400,
  },
  main: {
    display: 'flex',
    flex: 1,
    overflow: 'hidden',
    gap: 16,
    padding: 16,
    minHeight: 0,
  },
  leftPanel: {
    width: 300,
    flexShrink: 0,
    background: '#fff',
    borderRadius: 12,
    boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
    overflowY: 'auto',
    padding: '16px 0',
    minHeight: 0,
  },
  rightContent: {
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  card: {
    background: '#fff',
    borderRadius: 12,
    boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
    padding: 20,
  },
  section: {
    padding: '12px 16px',
    borderBottom: '1px solid #f0f0f0',
  },
  sectionTitle: {
    margin: '0 0 12px',
    fontSize: 14,
    fontWeight: 600,
    color: '#2d3748',
  },
  modeSwitch: {
    display: 'flex',
    gap: 8,
  },
  modeBtn: {
    flex: 1,
    padding: '10px 8px',
    border: '1px solid #e2e8f0',
    borderRadius: 8,
    background: '#f7fafc',
    cursor: 'pointer',
    fontSize: 13,
    fontWeight: 500,
    color: '#4a5568',
    transition: 'all 0.2s',
  },
  modeBtnActive: {
    background: '#3182ce',
    color: '#fff',
    borderColor: '#3182ce',
  },
  uploadArea: {
    border: '2px dashed #cbd5e0',
    borderRadius: 8,
    padding: '16px 12px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'border-color 0.2s',
  },
  uploadLabel: {
    cursor: 'pointer',
    fontSize: 13,
    color: '#4a5568',
  },
  inputGroup: {
    marginBottom: 10,
  },
  label: {
    display: 'block',
    fontSize: 12,
    fontWeight: 500,
    color: '#4a5568',
    marginBottom: 4,
  },
  inputWrap: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  input: {
    flex: 1,
    padding: '7px 10px',
    border: '1px solid #e2e8f0',
    borderRadius: 6,
    fontSize: 13,
    outline: 'none',
    transition: 'border-color 0.2s',
  },
  unit: {
    fontSize: 12,
    color: '#a0aec0',
    minWidth: 20,
  },
  btn: {
    padding: '10px 16px',
    borderRadius: 8,
    border: 'none',
    cursor: 'pointer',
    fontSize: 13,
    fontWeight: 600,
    transition: 'all 0.2s',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  btnPrimary: {
    background: '#3182ce',
    color: '#fff',
    boxShadow: '0 2px 4px rgba(49,130,206,0.3)',
  },
  btnDanger: {
    background: '#e53e3e',
    color: '#fff',
    boxShadow: '0 2px 4px rgba(229,62,62,0.3)',
    padding: '10px 16px',
  },
  btnSecondary: {
    background: '#edf2f7',
    color: '#4a5568',
    border: '1px solid #e2e8f0',
  },
  btnLarge: {
    padding: '14px 20px',
    fontSize: 14,
  },
  agentResult: {
    marginTop: 12,
    padding: 12,
    background: '#f7fafc',
    borderRadius: 8,
    border: '1px solid #e2e8f0',
  },
  agentBadge: (status) => ({
    display: 'inline-block',
    padding: '3px 10px',
    borderRadius: 12,
    fontSize: 11,
    fontWeight: 600,
    background: status === 'success' ? '#c6f6d5' : '#fed7d7',
    color: status === 'success' ? '#276749' : '#c53030',
  }),
  codeBlock: {
    fontSize: 10,
    background: '#edf2f7',
    padding: 8,
    borderRadius: 6,
    overflow: 'auto',
    maxHeight: 150,
    marginTop: 6,
  },
  progressBox: {
    padding: '12px 16px',
  },
  logContainer: {
    flex: 1,
    overflowY: 'auto',
    background: '#1a202c',
    borderRadius: 8,
    padding: 8,
    fontFamily: "'SF Mono', 'Fira Code', 'Consolas', monospace",
    fontSize: 11,
    lineHeight: '18px',
  },
  logLine: {
    display: 'flex',
    gap: 8,
    whiteSpace: 'nowrap',
  },
  logTime: {
    color: '#718096',
    minWidth: 70,
    flexShrink: 0,
  },
  logMsg: {
    color: '#e2e8f0',
    wordBreak: 'break-all',
    whiteSpace: 'normal',
  },
  canvasHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 12,
  },
  canvasTitle: {
    margin: 0,
    fontSize: 16,
    fontWeight: 600,
    color: '#2d3748',
  },
  canvasHint: {
    fontSize: 11,
    color: '#a0aec0',
  },
  canvasRow: {
    display: 'flex',
    gap: 16,
    alignItems: 'flex-start',
  },
  logPanel: {
    flex: '0 0 280px',
    maxHeight: 500,
    background: '#1a202c',
    borderRadius: 8,
    padding: 12,
    display: 'flex',
    flexDirection: 'column',
  },
  geometryInfo: {
    marginTop: 16,
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 8,
    width: '100%',
    maxWidth: 500,
  },
  infoRow: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '6px 10px',
    background: '#f7fafc',
    borderRadius: 6,
    fontSize: 12,
  },
  infoLabel: {
    color: '#718096',
    fontWeight: 500,
  },
  infoValue: {
    color: '#2d3748',
    fontFamily: 'monospace',
  },
  emptyResult: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '40px 20px',
    textAlign: 'center',
  },
  resultImageWrap: {
    borderRadius: 8,
    overflow: 'hidden',
    border: '1px solid #e2e8f0',
    background: '#fafafa',
    position: 'relative',
    cursor: 'pointer',
  },
  resultImage: {
    width: '100%',
    display: 'block',
  },
  resultOverlay: {
    position: 'absolute',
    inset: 0,
    background: 'rgba(0,0,0,0.45)',
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 16,
    fontWeight: 600,
    opacity: 0,
    transition: 'opacity 0.2s',
    pointerEvents: 'none',
  },
  lightbox: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0,0,0,0.85)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 9999,
    cursor: 'pointer',
  },
  lightboxImage: {
    maxWidth: '90vw',
    maxHeight: '90vh',
    borderRadius: 8,
    boxShadow: '0 4px 24px rgba(0,0,0,0.5)',
  },
  lossBox: {
    marginTop: 12,
    padding: '10px 14px',
    background: '#f0fff4',
    borderRadius: 8,
    border: '1px solid #c6f6d5',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  lossLabel: {
    fontSize: 12,
    color: '#276749',
    fontWeight: 500,
  },
  lossValue: {
    fontSize: 14,
    fontWeight: 600,
    color: '#22543d',
    fontFamily: 'monospace',
  },
  downloadRow: {
    display: 'flex',
    gap: 8,
    marginTop: 12,
  },
};

export default App;
