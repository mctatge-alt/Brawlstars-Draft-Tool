"use client";
import { useState } from "react";

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const fmt = (z: number) => (z >= 0 ? "+" : "−") + Math.abs(z).toFixed(2);

/**
 * Drag the logit. The point of the widget is the third readout: P(A) + P(B) reads 1.000 at
 * every position of the slider, because the architecture makes it an identity rather than
 * something the optimizer has to approximate.
 */
export default function MirrorTool() {
  const [z, setZ] = useState(0.8);
  const pa = sigmoid(z), pb = sigmoid(-z);

  return (
    <div className="mirror-tool">
      <div className="mirror-bars">
        <div className="mb-row">
          <div className="mb-label tA">P(A wins)</div>
          <div className="mb-track"><div className="mb-fill mb-a" style={{ width: `${(pa * 100).toFixed(2)}%` }} /></div>
          <div className="mb-val tA mono">{pa.toFixed(3)}</div>
        </div>
        <div className="mb-row">
          <div className="mb-label tB">P(B wins)</div>
          <div className="mb-track"><div className="mb-fill mb-b" style={{ width: `${(pb * 100).toFixed(2)}%` }} /></div>
          <div className="mb-val tB mono">{pb.toFixed(3)}</div>
        </div>
      </div>
      <label className="slider-wrap">
        <span className="tiny-lab">logit &#8467; &nbsp;<b className="mono">{fmt(z)}</b></span>
        <input type="range" min={-3} max={3} step={0.01} value={z}
          onChange={(e) => setZ(parseFloat(e.target.value))} aria-label="Logit value" />
      </label>
      <div className="sum-readout">
        <span className="tiny-lab">P(A) + P(B)</span>
        <span className="sum-val mono green">{(pa + pb).toFixed(3)}</span>
        <span className="tiny-lab dim">always, for every input</span>
      </div>
    </div>
  );
}
