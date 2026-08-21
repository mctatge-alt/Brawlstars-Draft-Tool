"use client";
import { useState } from "react";

// Per-state held-out metrics, from docs/metrics.json (embedding_partial.states). Only the six
// states the evaluation reports separately are listed; the other eight are trained on but not
// broken out.
const METRICS: Record<string, { ll: string; auc: string; ece: string; edge: string }> = {
  "1v0": { ll: "0.6908", auc: "0.538", ece: "0.010", edge: "0.029" },
  "1v1": { ll: "0.6870", auc: "0.561", ece: "0.010", edge: "0.044" },
  "2v1": { ll: "0.6839", auc: "0.576", ece: "0.010", edge: "0.058" },
  "2v2": { ll: "0.6781", auc: "0.595", ece: "0.010", edge: "0.070" },
  "3v2": { ll: "0.6742", auc: "0.608", ece: "0.013", edge: "0.082" },
  "3v3": { ll: "0.6674", auc: "0.625", ece: "0.009", edge: "0.093" },
};

type Kind = "full" | "none" | "mask";
const kindOf = (a: number, b: number): Kind => (a === 3 && b === 3 ? "full" : a === 0 && b === 0 ? "none" : "mask");

const ROLE: Record<Kind, React.ReactNode> = {
  full: <>Kept intact on <b>70%</b> of training rows. This is the full-comp objective, and the state the publish gate is measured on.</>,
  none: <>Never trained. Antisymmetry drives this logit to exactly zero, so there is no gradient to follow &mdash; and it still returns <b>0.500</b> at inference, for free.</>,
  mask: <>One of the <b>14</b> masked states. Together they take 30% of rows, drawn uniformly &mdash; roughly 2.1% each &mdash; and re-drawn from scratch every epoch.</>,
};

function Slots({ k, side }: { k: number; side: "a" | "b" }) {
  return (
    <>
      {[0, 1, 2].map((i) => (
        <div key={i} className={i < k ? `slot known-${side}` : "slot"}>{i < k ? "PICKED" : "MASK"}</div>
      ))}
    </>
  );
}

/** The 4x4 board of draft states the masking schedule draws from; click one to see what it means. */
export default function StateGrid() {
  const [[ka, kb], setSel] = useState<[number, number]>([2, 1]);
  const key = `${ka}v${kb}`;
  const m = METRICS[key];

  return (
    <div className="mask-layout">
      <div className="panel">
        <div className="kicker">Draft states seen in training</div>
        <div className="grid-axes">
          <div className="axis-y"><span>known picks &mdash; team A</span></div>
          <div>
            <div className="axis-x">known picks &mdash; team B</div>
            <div className="state-grid" role="group" aria-label="Draft state grid">
              {[0, 1, 2, 3].map((a) => [0, 1, 2, 3].map((b) => {
                const k = kindOf(a, b), sel = a === ka && b === kb;
                return (
                  <button key={`${a}v${b}`} type="button" aria-pressed={sel}
                    className={`cell${k === "full" ? " full" : k === "none" ? " none" : ""}${sel ? " sel" : ""}`}
                    onClick={() => setSel([a, b])}>{a}v{b}</button>
                );
              }))}
            </div>
          </div>
        </div>
        <div className="grid-legend">
          <span><i className="sw sw-mask" /> masked mixture &mdash; 30%, split evenly over 14 states</span>
          <span><i className="sw sw-full" /> full 3v3 &mdash; 70% of rows</span>
          <span><i className="sw sw-none" /> excluded &mdash; zero gradient</span>
        </div>
      </div>

      <div className="panel panel-2 state-detail" aria-live="polite">
        <div>
          <div className="kicker">Selected state</div>
          <div className="sd-title">{ka} v {kb}</div>
        </div>
        <div className="sd-board">
          <div className="sd-team"><span className="who tA">Team A</span><Slots k={ka} side="a" /></div>
          <div className="sd-team"><span className="who tB">Team B</span><Slots k={kb} side="b" /></div>
        </div>
        <p className="sd-role">{ROLE[kindOf(ka, kb)]}</p>
        {m ? (
          <div className="sd-metrics">
            <div><div className="k">Log-loss</div><div className="v">{m.ll}</div></div>
            <div><div className="k">AUC</div><div className="v">{m.auc}</div></div>
            <div><div className="k">ECE</div><div className="v green">{m.ece}</div></div>
            <div><div className="k">Edge</div><div className="v">{m.edge}</div></div>
          </div>
        ) : kindOf(ka, kb) === "mask" ? (
          <p className="sd-empty">Trained on, but not separately reported: the published evaluation measures six
            states &mdash; 1v0, 1v1, 2v1, 2v2, 3v2 and 3v3.</p>
        ) : (
          // The empty board is the one state that is genuinely never trained, so the
          // "trained on, but not reported" copy above would contradict its own role text.
          <p className="sd-empty">No held-out metrics, and none to have: this state never enters training, and
            the logit is exactly zero by construction.</p>
        )}
      </div>
    </div>
  );
}
