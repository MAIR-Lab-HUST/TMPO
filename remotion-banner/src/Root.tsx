import "./index.css";
import {Composition as RemotionComposition} from "remotion";
import {Composition} from "./Composition";
export const RemotionRoot:React.FC=()=> <RemotionComposition id="RewardDistribution" component={Composition} durationInFrames={96} fps={30} width={1200} height={420}/>;
