use anyhow::{anyhow, Context, Error as AnyhowError, Result};
use axum::{
    extract::{rejection::JsonRejection, Json, State},
    response::IntoResponse,
};
use futures::future::join_all;
use mistralrs_core::{
    Constraint, MistralRs, NormalRequest, Request, RequestMessage, Response, SamplingParams,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Receiver;
use utoipa::ToSchema;

use crate::{
    handler_core::{
        base_process_non_streaming_response, create_response_channel, openai_error_from_error,
        send_request_with_model, ApiError, ApiErrorKind, ModelErrorMessage,
    },
    types::{ExtractedMistralRsState, SharedMistralRsState},
    util::{parse_image_url_for_server, validate_model_name},
};

const IMAGE_PLACEHOLDER: &str = "{img}";
const IMAGE_TOKENS: &str = "<|vision_start|><|image_pad|><|vision_end|>";
const LABELS: [&str; 3] = ["contradiction", "entailment", "neutral"];

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ClassificationInput {
    pub premise: String,
    pub hypothesis: String,
    /// HTTP(S) URL or data URL. Put exactly one `{img}` marker in `premise`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ClassificationRequest {
    pub model: String,
    pub input: Vec<ClassificationInput>,
    #[serde(default)]
    pub truncate_sequence: bool,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ClassificationData {
    pub index: usize,
    pub label: &'static str,
    pub label_index: usize,
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ClassificationUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ClassificationResponse {
    pub object: &'static str,
    pub data: Vec<ClassificationData>,
    pub model: String,
    pub usage: ClassificationUsage,
}

pub enum ClassificationResponder {
    Json(ClassificationResponse),
    InternalError(AnyhowError),
    ValidationError(AnyhowError),
}

impl IntoResponse for ClassificationResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            Self::Json(response) => Json(response).into_response(),
            Self::InternalError(error) => {
                openai_error_from_error(error.as_ref(), ApiErrorKind::Internal)
            }
            Self::ValidationError(error) => {
                openai_error_from_error(error.as_ref(), ApiErrorKind::InvalidRequest)
            }
        }
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/classify",
    request_body = ClassificationRequest,
    responses((status = 200, description = "Classification logits and probabilities", body = ClassificationResponse))
)]
pub async fn classify(
    State(state): ExtractedMistralRsState,
    payload: Result<Json<ClassificationRequest>, JsonRejection>,
) -> ClassificationResponder {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return validation_error(AnyhowError::new(ApiError::from_json_rejection(error)));
        }
    };
    let repr =
        serde_json::to_string(&request).expect("Serialization of classification request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    if let Err(error) = validate_model_name(&request.model, state.clone()) {
        return validation_error(error.into());
    }
    if request.input.is_empty() {
        return validation_error(anyhow!(
            "input must contain at least one premise/hypothesis pair."
        ));
    }

    let mut inputs = Vec::with_capacity(request.input.len());
    for input in request.input {
        match prepare_classification_input(input).await {
            Ok(input) => inputs.push(input),
            Err(error) => return validation_error(error),
        }
    }

    let model_override = (request.model != "default").then_some(request.model.clone());
    let truncate_sequence = request.truncate_sequence;
    let futures = inputs.into_iter().map(|input| {
        let state = state.clone();
        let model_override = model_override.clone();
        async move {
            fetch_classification(state, input, model_override.as_deref(), truncate_sequence).await
        }
    });

    let mut data = Vec::new();
    let mut prompt_tokens = 0usize;
    let mut total_tokens = 0usize;
    for (index, result) in join_all(futures).await.into_iter().enumerate() {
        match result {
            Ok(output) => {
                let probabilities = softmax(&output.logits);
                let label_index = argmax(&probabilities);
                data.push(ClassificationData {
                    index,
                    label: LABELS[label_index],
                    label_index,
                    logits: output.logits,
                    probabilities,
                });
                prompt_tokens = prompt_tokens.saturating_add(output.prompt_tokens);
                total_tokens = total_tokens.saturating_add(output.total_tokens);
            }
            Err(error) => {
                MistralRs::maybe_log_error(state.clone(), error.as_ref());
                return internal_error(error);
            }
        }
    }

    let response = ClassificationResponse {
        object: "list",
        data,
        model: request.model,
        usage: ClassificationUsage {
            prompt_tokens,
            total_tokens,
        },
    };
    MistralRs::maybe_log_response(state, &response);
    ClassificationResponder::Json(response)
}

struct ClassificationOutput {
    logits: Vec<f32>,
    prompt_tokens: usize,
    total_tokens: usize,
}

struct PreparedClassificationInput {
    text: String,
    images: Vec<image::DynamicImage>,
}

async fn prepare_classification_input(
    input: ClassificationInput,
) -> Result<PreparedClassificationInput> {
    let placeholder_count = input.premise.matches(IMAGE_PLACEHOLDER).count();
    let images = match input.image {
        Some(source) => {
            if placeholder_count != 1 {
                anyhow::bail!(
                    "premise must contain exactly one `{IMAGE_PLACEHOLDER}` marker when image is set"
                );
            }
            vec![parse_image_url_for_server(&source)
                .await
                .with_context(|| format!("Failed to parse image resource: {source}"))?]
        }
        None => {
            if placeholder_count != 0 {
                anyhow::bail!(
                    "premise contains `{IMAGE_PLACEHOLDER}` but the request has no image"
                );
            }
            Vec::new()
        }
    };
    let premise = input.premise.replace(IMAGE_PLACEHOLDER, IMAGE_TOKENS);
    let text = format!(
        "Premise: {}\nHypothesis: {}",
        premise.trim(),
        input.hypothesis.trim()
    );
    Ok(PreparedClassificationInput { text, images })
}

async fn fetch_classification(
    state: SharedMistralRsState,
    input: PreparedClassificationInput,
    model_id: Option<&str>,
    truncate_sequence: bool,
) -> Result<ClassificationOutput> {
    let (tx, mut rx) = create_response_channel(Some(1));
    let request = Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        queued_at: None,
        messages: RequestMessage::Classification {
            text: input.text,
            images: input.images,
        },
        sampling_params: SamplingParams::deterministic(),
        seed: None,
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: None,
        constraint: Constraint::None,
        tool_choice: None,
        tools: None,
        logits_processors: None,
        return_raw_logits: true,
        web_search_options: None,
        enable_code_execution: false,
        enable_shell: false,
        shell_options: None,
        code_execution_permission: None,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_handler: None,
        agent_approval_notifier: None,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        model_id: model_id.map(str::to_string),
        adapter: None,
        truncate_sequence,
        session_id: None,
        files: None,
        input_files: Vec::new(),
    }));

    send_request_with_model(&state, request, model_id)
        .await
        .context("Failed to dispatch classification request")?;
    process_classification_response(&mut rx, state).await
}

async fn process_classification_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
) -> Result<ClassificationOutput> {
    base_process_non_streaming_response(
        rx,
        state.clone(),
        |_, response| match response {
            Response::Raw {
                logits_chunks,
                tokens,
            } => {
                let logits = logits_chunks
                    .last()
                    .ok_or_else(|| anyhow!("classification returned no logits"))?
                    .to_dtype(candle_core::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                if logits.len() != LABELS.len() {
                    anyhow::bail!(
                        "classification returned {} logits; expected {}",
                        logits.len(),
                        LABELS.len()
                    );
                }
                let prompt_tokens = tokens.len();
                Ok(ClassificationOutput {
                    logits,
                    prompt_tokens,
                    total_tokens: prompt_tokens,
                })
            }
            Response::ValidationError(error) => Err(AnyhowError::new(ApiError::from_error(
                error.as_ref(),
                ApiErrorKind::InvalidRequest,
            ))),
            Response::InternalError(error) => {
                MistralRs::maybe_log_error(state.clone(), error.as_ref());
                Err(anyhow!(error))
            }
            Response::ModelError(message, _) => {
                MistralRs::maybe_log_error(state.clone(), &ModelErrorMessage(message));
                Err(AnyhowError::new(ApiError::model_error()))
            }
            _ => Err(anyhow!("unexpected response type for classification")),
        },
        |_, error| Err(anyhow!(error)),
    )
    .await
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = logits
        .iter()
        .map(|logit| (*logit - max).exp())
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f32>();
    for probability in &mut probabilities {
        *probability /= sum;
    }
    probabilities
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("classification output is non-empty")
}

fn validation_error(error: AnyhowError) -> ClassificationResponder {
    ClassificationResponder::ValidationError(error)
}

fn internal_error(error: AnyhowError) -> ClassificationResponder {
    ClassificationResponder::InternalError(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_stable_and_normalized() {
        let probabilities = softmax(&[1000.0, 1001.0, 999.0]);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&probabilities), 1);
    }
}
