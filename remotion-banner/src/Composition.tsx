import {AbsoluteFill, interpolate, useCurrentFrame} from "remotion";

const W = 1200;
const H = 420;
const BASE = 320;
type Peak = {x: number; height: number; width: number};
const start: Peak[] = [{x: 210, height: 62, width: 34}, {x: 430, height: 228, width: 23}, {x: 690, height: 78, width: 38}, {x: 930, height: 118, width: 32}];
const end: Peak[] = [{x: 210, height: 112, width: 42}, {x: 430, height: 164, width: 49}, {x: 690, height: 118, width: 44}, {x: 930, height: 98, width: 42}];
const ease = (t: number) => t * t * (3 - 2 * t);
const curve = (peaks: Peak[]) => Array.from({length: 180}, (_, i) => { const x = 90 + (1080 * i) / 179; const y = peaks.reduce((sum, peak) => sum + peak.height * Math.exp(-((x - peak.x) ** 2) / (2 * peak.width ** 2)), 5); return `${x.toFixed(1)},${(BASE - y).toFixed(1)}`; }).join(" ");

export const Composition: React.FC = () => {
  const frame = useCurrentFrame();
  const t = ease(interpolate(frame, [12, 78], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"}));
  const peaks = start.map((a, i) => ({x: a.x, height: a.height + (end[i].height - a.height) * t, width: a.width + (end[i].width - a.width) * t}));
  const points = curve(peaks);
  const area = `90,${BASE} ${points} 1110,${BASE}`;
  const labelOpacity = interpolate(frame, [55, 88], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});
  return <AbsoluteFill style={{background: "#fff", fontFamily: "Arial, sans-serif"}}><svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`}>
    <defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#5b6ee1" stopOpacity=".28" /><stop offset="1" stopColor="#5b6ee1" stopOpacity=".03" /></linearGradient><linearGradient id="line" x1="0" x2="1"><stop stopColor="#5367d9" /><stop offset="1" stopColor="#1f9d91" /></linearGradient></defs>
    <text x="90" y="57" fontSize="30" fontWeight="700" fill="#172033">TMPO</text><text x="90" y="87" fontSize="17" fill="#667085">From reward maximization to reward distribution matching</text>
    <text x="1110" y="57" textAnchor="end" fontSize="16" fill="#5367d9" opacity={1 - labelOpacity}>mode-seeking</text><text x="1110" y="57" textAnchor="end" fontSize="16" fill="#1f8b80" opacity={labelOpacity}>mode-covering</text>
    <line x1="90" y1={BASE} x2="1110" y2={BASE} stroke="#9aa5b5" strokeWidth="2" /><line x1="90" y1="145" x2="90" y2={BASE} stroke="#9aa5b5" strokeWidth="2" />
    <polygon points={area} fill="url(#fill)" /><polyline points={points} fill="none" stroke="url(#line)" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
    <text x="90" y="350" fontSize="15" fill="#667085">trajectory space τ</text><text x="600" y="392" textAnchor="middle" fontSize="20" fontWeight="600" fill="#344054" opacity={labelOpacity}>Diversity preserved across high-reward trajectories</text>
  </svg></AbsoluteFill>;
};
