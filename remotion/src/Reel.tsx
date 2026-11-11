import {loadFont} from '@remotion/google-fonts/Montserrat';
import {
  AbsoluteFill,
  Audio,
  Easing,
  interpolate,
  OffthreadVideo,
  Sequence,
  staticFile,
  useCurrentFrame,
} from 'remotion';

const {fontFamily} = loadFont('normal', {weights: ['500', '600', '700'], subsets: ['cyrillic', 'latin']});

export type Shot = {
  src: string;
  // frame inside src where this shot starts
  startFrom: number;
  from: number;
  duration: number;
  // slow push in or pull out; the zoom only ever enlarges, so no frame edge shows
  move: 'in' | 'out' | null;
};

export type Text = {
  text: string;
  from: number;
  duration: number;
  kind: 'title' | 'caption' | 'sticker' | 'subtitle' | 'step';
};

export type ReelProps = {
  fps: number;
  width: number;
  height: number;
  durationInFrames: number;
  audio: string | null;
  shots: Shot[];
  texts: Text[];
};

const MOVE_ZOOM = 0.12;
const PUNCH_ZOOM = 0.06;
const PUNCH_FRAMES = 6;

const ShotView = ({shot}: {shot: Shot}) => {
  const frame = useCurrentFrame();
  const t = interpolate(frame, [0, shot.duration], [0, 1], {
    extrapolateRight: 'clamp',
    easing: Easing.inOut(Easing.sin),
  });
  const drift = shot.move === 'in' ? MOVE_ZOOM * t : shot.move === 'out' ? MOVE_ZOOM * (1 - t) : 0;
  const punch = interpolate(frame, [0, PUNCH_FRAMES], [PUNCH_ZOOM, 0], {
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });
  return (
    <AbsoluteFill style={{overflow: 'hidden'}}>
      <OffthreadVideo
        src={staticFile(shot.src)}
        startFrom={shot.startFrom}
        muted
        style={{width: '100%', height: '100%', objectFit: 'cover', transform: `scale(${1 + drift + punch})`}}
      />
    </AbsoluteFill>
  );
};

// one type style for every line; only size and place change with the kind
const KIND = {
  title: {size: 76, weight: 700, spacing: 2, top: 820},
  sticker: {size: 44, weight: 500, spacing: 1, top: 1240},
  caption: {size: 30, weight: 600, spacing: 8, top: 1560},
  subtitle: {size: 58, weight: 700, spacing: 0, top: 1300},
  step: {size: 40, weight: 700, spacing: 6, top: 220},
} as const;

const MAX_WIDTH = 880;
const GLYPH_WIDTH = 0.62;

// shrink long subtitles so they wrap to two lines at most inside MAX_WIDTH
const fitSize = (text: string, size: number) =>
  Math.min(size, Math.floor((2 * MAX_WIDTH * 0.9) / (GLYPH_WIDTH * text.length)));

const TextLine = ({text, kind, duration}: {text: string; kind: Text['kind']; duration: number}) => {
  const frame = useCurrentFrame();
  const style = KIND[kind];
  const enter = kind === 'subtitle' ? 1 : interpolate(frame, [0, 14], [0, 1], {extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
  const leave = kind === 'caption' || kind === 'subtitle' ? 1 : interpolate(frame, [duration - 10, duration], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill style={{justifyContent: 'flex-start', alignItems: 'center', paddingTop: style.top}}>
      <div
        style={{
          fontFamily,
          fontWeight: style.weight,
          fontSize: kind === 'subtitle' ? fitSize(text, style.size) : style.size,
          letterSpacing: style.spacing,
          lineHeight: 1.15,
          color: 'white',
          textAlign: 'center',
          maxWidth: MAX_WIDTH,
          textWrap: 'balance',
          textShadow: kind === 'subtitle' ? '0 1px 4px rgba(0,0,0,0.9), 0 2px 18px rgba(0,0,0,0.6)' : '0 2px 18px rgba(0,0,0,0.55)',
          opacity: enter * leave,
          transform: `translateY(${(1 - enter) * 16}px)`,
        }}
      >
        {text}
      </div>
    </AbsoluteFill>
  );
};

export const Reel = ({shots, audio, texts}: ReelProps) => (
  <AbsoluteFill style={{backgroundColor: 'black'}}>
    {shots.map((shot) => (
      <Sequence key={shot.from} from={shot.from} durationInFrames={shot.duration}>
        <ShotView shot={shot} />
      </Sequence>
    ))}
    {audio ? <Audio src={staticFile(audio)} /> : null}
    {texts.map((t, i) => (
      <Sequence key={i} from={t.from} durationInFrames={t.duration}>
        <TextLine text={t.text} kind={t.kind} duration={t.duration} />
      </Sequence>
    ))}
  </AbsoluteFill>
);
