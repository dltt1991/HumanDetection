# Detection Box Tracking Stability Design

## Goal

Reduce visible person-box jitter in the macOS camera preview without buffering
or delaying video frames. Detection boxes may lag the current person position by
approximately two to four frames.

## Scope

Add lightweight temporal box tracking after PicoDet post-processing and before
drawing. Keep the current model, preprocessing, inference backend, NMS, and
camera capture flow unchanged. Do not add an external tracking dependency,
appearance embeddings, optical flow, or an RKNN implementation.

## Design

Introduce a small `BoxTracker` component in the core library. Each call accepts
the detections for the current frame and returns stabilized detections to draw.

Tracks are associated with current detections using greedy highest-IoU matching.
A pair is eligible when its IoU is at least `0.3`. Each detection and track may
be matched once. This is sufficient for the low-speed reversing-camera scenario
and keeps the implementation portable to Intel and ARM CPUs.

For a matched track, each box coordinate is updated with an exponential moving
average:

```
smoothed = 0.35 * detected + 0.65 * previous_smoothed
```

The score follows the current detection so the displayed confidence remains
meaningful. New detections create tracks immediately. An unmatched track keeps
its last smoothed box for up to three frames, then expires. This suppresses
single-frame dropouts while bounding stale-box lifetime.

The camera loop continues to read, infer, draw, and display the current frame.
No video-frame queue or delayed frame rendering is introduced; only the box
coordinates have temporal inertia.

## Interface And Configuration

`BoxTracker` owns track state and exposes one update method. The application
creates one tracker for the camera session and feeds it restored, frame-space
detections. Tracking therefore remains independent of model tensor coordinates.

Expose `--box-smoothing VALUE` with default `0.35`, constrained to `(0, 1]`.
`1.0` disables coordinate smoothing while retaining association and dropout
handling. Keep the IoU threshold and three-frame lifetime fixed until real-world
testing demonstrates a need to tune them.

## Error Handling

Reject non-finite or out-of-range smoothing values during CLI parsing. Ignore
degenerate detection boxes in the tracker so invalid geometry cannot persist
across frames.

## Testing

Add focused unit tests that verify:

- A stationary person's noisy boxes converge to a steadier box.
- A moving person's output uses the configured EMA update.
- Two separated people remain associated with their own tracks.
- A one-frame missed detection retains the box, while four misses remove it.
- Invalid smoothing CLI values are rejected through the executable test path.

Run the existing preprocessing, post-processing, model smoke, and help tests to
guard the complete macOS path. Final visual judgment still requires an
interactive camera session because automated tests cannot measure perceived
jitter from the attached USB camera.
