from avlm.data.media import interleave_frames_and_subtitles
from avlm.data.subtitles import load_subtitles
from avlm.data.video import IndexedVideoReader
from avlm.methods.adaq import AdaQArtifactSelector, normalize_option_scores
from avlm.models import load_adapter


def load_model(model_path, device="cuda:1", attention_backend="eager"):
    adapter = load_adapter(model_path, attention_backend=attention_backend,
        dtype="bfloat16", device_map={"": device}, local_files_only=True)
    adapter.model.eval().requires_grad_(False)
    return adapter


def select_frames(adaq_indices, frame_count):
    if frame_count <= 0 or not 1 <= len(adaq_indices) <= 64:
        raise ValueError("Expected a nonempty video and 1 to 64 AdaQ indices")
    if any(type(i) is not int or not 0 <= i < frame_count for i in adaq_indices):
        raise ValueError("AdaQ indices must be integers within the decoded video")
    if tuple(adaq_indices) != tuple(sorted(set(adaq_indices))):
        raise ValueError("AdaQ indices must be sorted and unique")
    anchors = tuple(sorted({min(((2 * i + 1) * frame_count) // 64,
                                frame_count - 1) for i in range(32)}))
    indices = tuple(sorted(set(adaq_indices) | set(anchors)))
    if len(indices) > 96:
        raise ValueError("Union evidence exceeds 96 frames")
    return indices, anchors


def svi_single_instance(view, adapter, selector):
    selection = selector.select(view)
    reader = IndexedVideoReader(view["video_path"])
    indices, anchors = select_frames(selection.frame_indices, reader.metadata.frame_count)
    cues = (load_subtitles(view["subtitle_path"], offset=float(view["subtitle_offset"]))
            if view["subtitle_path"] else [])
    frames = reader.decode_indices(indices)
    events = interleave_frames_and_subtitles(frames, cues)
    batch = adapter.prepare_interleaved_inputs(
        events=events,
        segment_header="AdaQ64 + Uniform32 Union96 | 0.00-%.2f s" % float(view["duration"]),
        question=view["question"], options=view["choices"],
        min_pixels=1024, max_pixels=50176)
    answer = adapter.generate_prepared_choice_constrained(batch, options=view["choices"])
    scores, probabilities = normalize_option_scores(answer.option_scores,
                                                    option_count=len(view["choices"]))
    return {
        "method": "SVI-single instance",
        "sample_id": view["sample_id"],
        "video_id": view["video_id"],
        "prediction": answer.choice,
        "invalid_answer": answer.choice is None,
        "raw_output": answer.text,
        "option_scores": scores,
        "option_probabilities": probabilities,
        "frame_indices": list(indices),
        "adaq_frame_indices": list(selection.frame_indices),
        "uniform_anchor_frame_indices": list(anchors),
        "frames": len(indices),
        "subtitle_count": sum(event.kind == "subtitle" for event in events),
        "answer_forward_calls": 1,
    }
