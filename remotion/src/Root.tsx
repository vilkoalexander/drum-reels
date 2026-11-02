import {Composition} from 'remotion';
import {Reel, ReelProps} from './Reel';

const sample: ReelProps = {
  fps: 30,
  width: 1080,
  height: 1920,
  durationInFrames: 90,
  audio: null,
  shots: [],
  texts: [{text: 'Title here', from: 0, duration: 90, kind: 'title'}],
};

export const Root = () => (
  <Composition
    id="Reel"
    component={Reel}
    defaultProps={sample}
    fps={sample.fps}
    width={sample.width}
    height={sample.height}
    durationInFrames={sample.durationInFrames}
    calculateMetadata={({props}) => ({
      fps: props.fps,
      width: props.width,
      height: props.height,
      durationInFrames: props.durationInFrames,
    })}
  />
);
