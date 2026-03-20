"""Unsloth Vision-Language Model loader."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sita.core.base_model import BaseModelLoader
from sita.core.config import ModelConfig
from sita.core.registry import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Qwen-VL ChatML template (vision-aware)
#
# This is the official Qwen 2.5/3.5 VL chat template that natively handles
# multimodal content lists (images, videos), tool calls, and <think> tags.
# Stored as a module-level constant to avoid string-escaping issues.
# ---------------------------------------------------------------------------
_QWEN_VL_CHAT_TEMPLATE = """\
{%- set image_count = namespace(value=0) -%}
{%- set video_count = namespace(value=0) -%}
{%- macro render_content(content, do_vision_count, is_system_content=false) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- elif content is iterable and content is not mapping -%}
        {%- for item in content -%}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' -%}
                {%- if is_system_content -%}
                    {{- raise_exception('System message cannot contain images.') -}}
                {%- endif -%}
                {%- if do_vision_count -%}
                    {%- set image_count.value = image_count.value + 1 -%}
                {%- endif -%}
                {%- if add_vision_id -%}
                    {{- 'Picture ' ~ image_count.value ~ ': ' -}}
                {%- endif -%}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' -}}
            {%- elif 'video' in item or item.type == 'video' -%}
                {%- if is_system_content -%}
                    {{- raise_exception('System message cannot contain videos.') -}}
                {%- endif -%}
                {%- if do_vision_count -%}
                    {%- set video_count.value = video_count.value + 1 -%}
                {%- endif -%}
                {%- if add_vision_id -%}
                    {{- 'Video ' ~ video_count.value ~ ': ' -}}
                {%- endif -%}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' -}}
            {%- elif 'text' in item -%}
                {{- item.text -}}
            {%- else -%}
                {{- raise_exception('Unexpected item type in content.') -}}
            {%- endif -%}
        {%- endfor -%}
    {%- elif content is none or content is undefined -%}
        {{- '' -}}
    {%- else -%}
        {{- raise_exception('Unexpected content type.') -}}
    {%- endif -%}
{%- endmacro -%}
{%- if not messages -%}
    {{- raise_exception('No messages provided.') -}}
{%- endif -%}
{%- if tools and tools is iterable and tools is not mapping -%}
    {{- '<|im_start|>system\n' -}}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" -}}
    {%- for tool in tools -%}
        {{- "\n" -}}
        {{- tool | tojson -}}
    {%- endfor -%}
    {{- "\n</tools>" -}}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' -}}
    {%- if messages[0].role == 'system' -%}
        {%- set content = render_content(messages[0].content, false, true)|trim -%}
        {%- if content -%}
            {{- '\n\n' + content -}}
        {%- endif -%}
    {%- endif -%}
    {{- '<|im_end|>\n' -}}
{%- else -%}
    {%- if messages[0].role == 'system' -%}
        {%- set content = render_content(messages[0].content, false, true)|trim -%}
        {{- '<|im_start|>system\n' + content + '<|im_end|>\n' -}}
    {%- endif -%}
{%- endif -%}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) -%}
{%- for message in messages[::-1] -%}
    {%- set index = (messages|length - 1) - loop.index0 -%}
    {%- if ns.multi_step_tool and message.role == "user" -%}
        {%- set content = render_content(message.content, false)|trim -%}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) -%}
            {%- set ns.multi_step_tool = false -%}
            {%- set ns.last_query_index = index -%}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if ns.multi_step_tool -%}
    {{- raise_exception('No user query found in messages.') -}}
{%- endif -%}
{%- for message in messages -%}
    {%- set content = render_content(message.content, true)|trim -%}
    {%- if message.role == "system" -%}
        {%- if not loop.first -%}
            {{- raise_exception('System message must be at the beginning.') -}}
        {%- endif -%}
    {%- elif message.role == "user" -%}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' -}}
    {%- elif message.role == "assistant" -%}
        {%- set reasoning_content = '' -%}
        {%- if message.reasoning_content is string -%}
            {%- set reasoning_content = message.reasoning_content -%}
        {%- else -%}
            {%- if '</think>' in content -%}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') -%}
                {%- set content = content.split('</think>')[-1].lstrip('\n') -%}
            {%- endif -%}
        {%- endif -%}
        {%- set reasoning_content = reasoning_content|trim -%}
        {%- if loop.index0 > ns.last_query_index -%}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content -}}
        {%- else -%}
            {{- '<|im_start|>' + message.role + '\n' + content -}}
        {%- endif -%}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping -%}
            {%- for tool_call in message.tool_calls -%}
                {%- if tool_call.function is defined -%}
                    {%- set tool_call = tool_call.function -%}
                {%- endif -%}
                {%- if loop.first -%}
                    {%- if content|trim -%}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' -}}
                    {%- else -%}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' -}}
                    {%- endif -%}
                {%- else -%}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' -}}
                {%- endif -%}
                {%- if tool_call.arguments is mapping -%}
                    {%- for args_name in tool_call.arguments -%}
                        {%- set args_value = tool_call.arguments[args_name] -%}
                        {{- '<parameter=' + args_name + '>\n' -}}
                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string -%}
                        {{- args_value -}}
                        {{- '\n</parameter>\n' -}}
                    {%- endfor -%}
                {%- endif -%}
                {{- '</function>\n</tool_call>' -}}
            {%- endfor -%}
        {%- endif -%}
        {{- '<|im_end|>\n' -}}
    {%- elif message.role == "tool" -%}
        {%- if loop.previtem and loop.previtem.role != "tool" -%}
            {{- '<|im_start|>user' -}}
        {%- endif -%}
        {{- '\n<tool_response>\n' -}}
        {{- content -}}
        {{- '\n</tool_response>' -}}
        {%- if not loop.last and loop.nextitem.role != "tool" -%}
            {{- '<|im_end|>\n' -}}
        {%- elif loop.last -%}
            {{- '<|im_end|>\n' -}}
        {%- endif -%}
    {%- else -%}
        {{- raise_exception('Unexpected message role.') -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n' -}}
    {%- if enable_thinking is defined and enable_thinking is true -%}
        {{- '<think>\n' -}}
    {%- else -%}
        {{- '<think>\n\n</think>\n\n' -}}
    {%- endif -%}
{%- endif -%}"""


@MODEL_REGISTRY.register("unsloth_vlm")
class UnslothVLMLoader(BaseModelLoader):
    """Load an Unsloth FastVisionModel + Tokenizer.

    Uses ``FastVisionModel.from_pretrained`` which returns (model, tokenizer).
    Config kwargs are forwarded directly, so you can pass ``max_seq_length``,
    ``load_in_4bit``, ``load_in_16bit``, ``dtype``, etc.

    Example YAML::

        model:
          name: unsloth_vlm
          pretrained: unsloth/Qwen3.5-0.8B
          kwargs:
            max_seq_length: 2048
            load_in_4bit: true
            chat_template: chatml
    """

    def load(self, config: ModelConfig) -> tuple[nn.Module, Any]:
        try:
            from unsloth import FastVisionModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Please install it using instructions from "
                "https://github.com/unslothai/unsloth"
            )

        kwargs = dict(config.kwargs)

        chat_template = kwargs.pop("chat_template", None)

        if "dtype" in kwargs and kwargs["dtype"] in ("None", None):
            kwargs.pop("dtype", None)

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=config.pretrained,
            **kwargs,
        )

        if chat_template is not None:
            from unsloth.chat_templates import get_chat_template
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=chat_template,
            )
            # Override with the comprehensive Qwen-VL vision-aware template
            if chat_template == "chatml":
                tokenizer.chat_template = _QWEN_VL_CHAT_TEMPLATE

        return model, tokenizer
