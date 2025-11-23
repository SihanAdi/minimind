"""
BPE 最初是一种数据压缩算法，后来被非常成功地应用于自然语言处理（NLP）的分词任务中。
* 核心思想：从最基础的单元（比如字母或字符）开始，迭代地将最频繁共现的一对单元（a pair）合并成一个新的、更大的单元。
* 在 NLP 中的应用（例如在 GPT-2 中）：

    1. 初始化：将单词拆分成字符（例如，"low" -> ['l', 'o', 'w']），并在单词末尾添加一个特殊的结束符号（如 </w>）以区分单词边界。"lower" -> ['l', 'o', 'w', 'e', 'r', '</w>']。
    2. 构建词表：统计文本中所有字符对的出现频率。
    3. 迭代合并：找到频率最高的字符对（比如 ('e', 'r')），将它们合并成一个新的符号 'er'。然后更新词表，并将这个合并操作应用到所有单词中。接着继续找下一个最频繁的对（可能是 ('er', '</w>')），合并成 'er</w>'）。如此反复。
    4. 停止：当合并操作进行了预定的次数（即词表达到预定的大小时）停止。

* 优点：
    * 能有效地在词频和字符级之间取得平衡，很好地处理未登录词（OOV）问题。
    * 比单纯按空格分词能捕获更多的语言结构信息。

* 缺点：
    * 对于多语言文本，字符集可能非常庞大（如中文、日文的汉字），导致初始词表巨大。
    * 无法保证基础单元（字符）本身是有效的 UTF-8 字符，可能在多语言环境中出现问题。
"""
import random
import json
from transformers import AutoTokenizer
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
import os

random.seed(42)

def read_texts_from_jsonl(file_path):
    """
    input jsonl file example:
    {"text": "<|im_start|>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。<|im_end|> "}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            yield data["text"]

def train_tokenizer(file_path):
    # 创建BPE分词器
    tokenizer = Tokenizer(models.BPE())
    
    # 设置预分词器：先将文本拆分为更小单元，为BPE算法做准备
    # 将文本转换为UTF-8字节序列; 普通ASCII字符保持原样; 不可打印字符处理：用特定的转义序列表示
    # add_prefix_space: 在每个词的开头添加空格，是否添加取决于是否将空格视为单词的一部分
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>"
    ]

    # 创建BPE训练器
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()    # 初始字母表
    )
    print(f"ByteLevel.alphabet: {pre_tokenizers.ByteLevel.alphabet()}")

    trainset = read_texts_from_jsonl(file_path)

    # 训练 tokenizer
    # 训练过程
    # 1. 预处理：使用预分词器拆分文本; 2. 统计频率：计算所有字符和字符对的频率; 3. 合并操作：迭代合并最常见的字符对; 4. 构建词汇表：直到达到vocab_size限制
    tokenizer.train_from_iterator(trainset, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查 special token 索引
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # 保存 tokenizer
    tokenizer_dir = "../model_weights/"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))    # 保存tokenizer的配置信息和词汇表
    tokenizer.model.save(tokenizer_dir)    # 保存分词模型的具体数据文件

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")


def evaluate_tokenizer(messages, tokenizer_dir="../model_weights/"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(prompt)
    #  model_inputs.keys(): {'input_ids', 'token_type_ids', 'attention_mask'})
    print(len(model_inputs["input_ids"]))
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == prompt)




if __name__ == '__main__':
    train_tokenizer('../dataset/pretrain_hq.jsonl')
    messages = [
        {"role": "system", "content": "你是一个优秀的AI助手，总是给我正确的回应！"},
        {"role": "user", "content": '你是谁？'},
        {"role": "assistant", "content": '我是一个AI小助手'}
    ]
    evaluate_tokenizer(messages)