# Lab 7 — Integration Retrospective

Ekaterina Shigarova · IIKb-24 · 21 November 2026

## Outcome

I connected the fine-tuned EfficientNet-B0 ONNX model to a shared quality-inference pipeline and a Streamlit website deployed at https://document-quality-classification.onrender.com. The model is loaded once and reused. The application accepts good images, rejects poor-quality images with retake advice, and returns structured JSON errors for invalid inputs. Following the instructor's clarification, document-type classification remains deferred to next semester.

## What failed and how I handled it

The first deployment route, Streamlit Community Cloud, stopped at authentication with `policy_denied` and an access-blocked message. The screen did not establish the cause. I switched to a Render web service while keeping the same application and model. This separated the hosting problem from the model's already working local inference.

Moving from training code to a deployable application required explicit paths, dependencies and output semantics. The ONNX model outputs `P(good)`, while Task 5B selected a threshold for `P(bad)`. The pipeline therefore computes `1 - P(good)` and applies the strict `> 0.35` rule. A boundary glare fixture verifies that changing the configured threshold changes the decision. Model paths are relative to `config.yaml`, avoiding dependence on the Windows project path.

Successful predictions alone did not demonstrate reliable integration. I added checks for corrupted PNGs, a text file renamed as JPG, missing models, JSON types and model-session reuse. Invalid files produce readable UI messages and error JSON instead of a traceback. The public website also handled a corrupted file without interrupting subsequent use.

## What the evidence establishes

All **10 pytest tests passed**. The batch run produced **10 accepted, 10 rejected and 2 errors**, matching the fixtures. Local mean CPU processing time was **19.11 ms** over 20 timed images after 20 warm-up requests, below the 500 ms requirement. Public checks confirmed good/bad decisions and invalid-file handling. Two hosted requests showed 176.5 ms and 357.4 ms, illustrating that laptop timing should not be presented as cloud performance.

The integration fixtures came from validation and are not a new accuracy benchmark. Historical 99.64% synthetic test accuracy used threshold 0.50; the deployed threshold is 0.35. I have not established full test accuracy at 0.35 or real-photo accuracy. The demo recording documents live behavior but does not replace the live classroom demonstration.

## What I would change next

I would validate the final operating point on an independent, consented real-photo dataset, with results by document type and defect. I would measure hosted warm-request and cold-start latency separately, add a complete dependency lock and automated deployment tests, and test concurrent uploads under the cloud resource limit. I would also calibrate confidence and add an explicit ambiguous-case review path. The next-semester document classifier can use the existing JSON fields without changing the quality gate's contract.
