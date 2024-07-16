import os
from typing import List
from typing import Union

import openai
import tiktoken
from dotenv import load_dotenv
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()  # read local .env file
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_TOKENS_PER_CHUNK = (
    500  # if text is more than this many tokens, we'll break it up into
)
# discrete chunks to translate one chunk at a time


def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    json_mode: bool = False,
) -> Union[str, dict]:
    """
        Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model (str, optional): The name of the OpenAI model to use for generating the completion.
            Defaults to "gpt-3.5-turbo".
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
            Defaults to 0.3.
        json_mode (bool, optional): Whether to return the response in JSON format.
            Defaults to False.

    Returns:
        Union[str, dict]: The generated completion.
            If json_mode is True, returns the complete API response as a dictionary.
            If json_mode is False, returns the generated text as a string.
    """

    if json_mode:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


def one_chunk_initial_translation(
    source_lang: str, target_lang: str, source_text: str
) -> str:
    """
    Translate the entire text as one chunk using an LLM.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.

    Returns:
        str: The translated text.
    """

    system_message = f"你是语言专家，擅长将{source_lang}翻译为{target_lang}"

    translation_prompt = f"""下面三重引号(''')中的文本是{source_lang}，你需要将它翻译为{target_lang}。
除了翻译之外，不要提供任何解释或文本。
{source_lang}:'''{source_text}'''
{target_lang}:"""

    prompt = translation_prompt.format(source_text=source_text)

    translation = get_completion(prompt, system_message=system_message)

    return translation


def one_chunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    country: str = "",
) -> str:
    """
    Use an LLM to reflect on the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        country (str): Country specified for target language.

    Returns:
        str: The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.
    """

    system_message = f"你是语言专家，擅长将{source_lang}翻译为{target_lang}。 \
你将获得源文本及其翻译，您的目标是改进翻译。"

    if country != "":
        country_prompt = f"译文的最终风格和语气应与{target_lang}在{country}口语中的风格相匹配。"
    else:
        country_prompt = ""
        
    reflection_prompt = f"""你的任务是仔细阅读一篇源文本和一篇从{source_lang}到{target_lang}的译文，然后给出建设性的批评和有用的建议来改进译文。\
{country_prompt}

由XML标记<SOURCE_TEXT></SOURCE_TEXT>和<TRANSLATION></TRANSLATION>分隔的源文本和初始翻译如下：

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

写建议时，注意是否有办法根据以下几点提高译文质量 \n\
1. 准确性(是否有添加、误译、遗漏或未翻译文本的错误),\n\
2. 流畅(应用{target_lang}语法、拼写和标点规则，并避免不必要的重复,\n\
3. 风格(翻译应反映源文本的风格并考虑文化背景),\n\
4. 术语(通过确保术语使用一致并反映源文本域；并确保使用{target_lang}的等效成语)。\n\

写一份具体的、有用的和建设性的建议清单，以改进翻译。
每个建议都应针对翻译的一个特定部分。
只输出建议，不输出其他内容。"""


    prompt = reflection_prompt.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation_1=translation_1,
        country_prompt=country_prompt,
    )
    reflection = get_completion(prompt, system_message=system_message)
    return reflection


def one_chunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    reflection: str,
) -> str:
    """
    使用反射来改进翻译，将整个文本视为一个块。

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        reflection (str): Expert suggestions and constructive criticism for improving the translation.

    Returns:
        str: The improved translation based on the expert suggestions.
    """

    system_message = f"你是语言专家，从事{source_lang}到{target_lang}的翻译和编辑工作。"

    prompt = f"""你的任务是仔细阅读源文本、初始翻译和语言学专家家建议，充分考虑语言学专家家建议，然后重新编辑初始翻译，输出最终翻译。

源文本、初始翻译和语言学专家家建议由XML标记<SOURCE_TEXT></SOURCE_TEXT>、<TRANSLATION></TRANSLATION>和<EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS>分隔，内容如下：

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>


只输出新翻译，没有其他内容。"""

    translation_2 = get_completion(prompt, system_message)

    return translation_2


def one_chunk_translate_text(
    source_lang: str, target_lang: str, source_text: str, country: str = ""
) -> str:
    """
    Translate a single chunk of text from the source language to the target language.

    This function performs a two-step translation process:
    1. Get an initial translation of the source text.
    2. Reflect on the initial translation and generate an improved translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The text to be translated.
        country (str): Country specified for target language.
    Returns:
        str: The improved translation of the source text.
    """
    translation_1 = one_chunk_initial_translation(
        source_lang, target_lang, source_text
    )
    ic(f"translation_1:\n\n{translation_1}")
    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country
    )
    ic(f"reflection:\n\n{reflection}")
    translation_2 = one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection
    )

    return translation_2


def num_tokens_in_string(
    input_str: str, encoding_name: str = "cl100k_base"
) -> int:
    """
    Calculate the number of tokens in a given string using a specified encoding.

    Args:
        str (str): The input string to be tokenized.
        encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base",
            which is the most commonly used encoder (used by gpt-3.5-turbo).

    Returns:
        int: The number of tokens in the input string.

    Example:
        >>> text = "Hello, how are you?"
        >>> num_tokens = num_tokens_in_string(text)
        >>> print(num_tokens)
        5
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens


def multichunk_initial_translation(
    source_lang: str, target_lang: str, source_text_chunks: List[str]
) -> List[str]:
    """
    将文本从源语言翻译成目标语言.

    Args:
        source_lang（str）：文本的源语言。
        target_lang（str）：翻译的目标语言。
        source_text_chunks（List[str]）：要翻译的文本块列表。

    Returns:
        list[str]：翻译文本块的列表。
    """

    system_message = f"你是语言专家，擅长将{source_lang}翻译为{target_lang}。"

    translation_prompt = """您的任务是将提供的文本中<TRANSLATE_THIS></TRANSLATE_THIS>的内容，从{source_lang}翻译为{target_lang}。

我会给你提供三段文本，分别由XML标记<PREV></PREV>、<TRANSLATE_THIS></TRANSLATE_THIS>、<NEXT></NEXT>包裹，<TRANSLATE_THIS></TRANSLATE_THIS>包裹的是要翻译的文本，<PREV></PREV>包裹的是文本的前文，<NEXT></NEXT>包裹的是文本的后文.
内容如下：
<PREV>{prev}</PREV>
<TRANSLATE_THIS>{translate_this}</TRANSLATE_THIS>
<NEXT>{next}</NEXT>

你可以参考PREV和NEXT中的前后文，但不要翻译任何前后文。
强调一遍，你仅翻译XML标记<TRANSLATE_THIS></TRANSLATE_THIS>中的内容，除此之外不要输出任何内容。

"""
    ic(len(source_text_chunks))
    translation_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        # tagged_text = (
        #     "".join(source_text_chunks[0:i])
        #     + "<TRANSLATE_THIS>"
        #     + source_text_chunks[i]
        #     + "</TRANSLATE_THIS>"
        #     + "".join(source_text_chunks[i + 1 :])
        # )

        prompt = translation_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            prev=source_text_chunks[i - 1] if i > 0 else "",
            next=source_text_chunks[i + 1] if i < len(source_text_chunks) - 1 else "",
            translate_this=source_text_chunks[i],
        )

        translation = get_completion(prompt, system_message=system_message)
        translation_chunks.append(translation)

    return translation_chunks


def multichunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    country: str = "",
) -> List[str]:
    """
    为改进部分译文提供建设性的批评和建议。

    Args:
        source_lang（str）：文本的源语言。
        target_lang（str）：译文的目标语言。
        source_text_chunks（List[str]）：分成块的源文本。
        translation_1_chunks（List[str]）：源文本块对应的翻译块。
        country（str）：为目标语言指定的国家。

    Returns:
        List[str]: 包含改进每个翻译块的建议的反射列表。
    """

    system_message = f"你是一位专业的语言学家，擅长将{source_lang}翻译成{target_lang}。 \
你将获得源文本及其翻译，你的目标是改进翻译。"
    if country != "":
        country_prompt = f"译文的最终风格和语气应与{target_lang}在{country}口语中的风格相匹配。"
    else:
        country_prompt = ""
    reflection_prompt = """你的任务是仔细阅读一篇源文本和一篇从{{source_lang}}到{{target_lang}}的译文，然后给出建设性的批评和有用的建议来改进译文。
{country_prompt}

我会给你提供四段文本，分别由XML标记<PREV></PREV>、<TRANSLATE_THIS></TRANSLATE_THIS>、<NEXT></NEXT>、<TRANSLATION></TRANSLATION>包裹，<TRANSLATE_THIS></TRANSLATE_THIS>包裹的是要翻译的文本，<TRANSLATION></TRANSLATION>包裹的是翻译后的文本，<PREV></PREV>包裹的是原文的前文，<NEXT></NEXT>包裹的是原文的后文。
内容如下：
<PREV>{prev}</PREV>
<TRANSLATE_THIS>{translate_this}</TRANSLATE_THIS>
<NEXT>{next}</NEXT>

<TRANSLATION>{translation}</TRANSLATION>

写建议时，注意是否有办法根据以下几点提高译文质量 \n\
1. 准确性(是否有添加、误译、遗漏或未翻译文本的错误),\n\
2. 流畅(应用{{target_lang}}语法、拼写和标点规则，并避免不必要的重复,\n\
3. 风格(翻译应反映源文本的风格并考虑文化背景),\n\
4. 术语(通过确保术语使用一致并反映源文本域；并确保使用{{target_lang}}的等效成语)。\n\

写一份具体的、有用的和建设性的建议清单，以改进翻译。
每个建议都应针对翻译的一个特定部分。
只输出建议，不输出其他内容。
"""


    reflection_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        # tagged_text = (
        #     "".join(source_text_chunks[0:i])
        #     + "<TRANSLATE_THIS>"
        #     + source_text_chunks[i]
        #     + "</TRANSLATE_THIS>"
        #     + "".join(source_text_chunks[i + 1 :])
        # )
        prompt = reflection_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            prev=source_text_chunks[i - 1] if i > 0 else "",
            next=source_text_chunks[i + 1] if i < len(source_text_chunks) - 1 else "",
            translate_this=source_text_chunks[i],
            translation=translation_1_chunks[i],
            country_prompt=country_prompt,
        )

        reflection = get_completion(prompt, system_message=system_message)
        reflection_chunks.append(reflection)

    return reflection_chunks


def multichunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    reflection_chunks: List[str],
) -> List[str]:
    """
    通过考虑专家建议改进文本从源语言到目标语言的翻译。

    Args:
        source_lang (str): 文本的源语言。
        target_lang (str): 翻译的目标语言。
        source_text_chunks (List[str]): 将源文本划分为块。
        translation_1_chunks (List[str]): 每个块的初始翻译。
        reflection_chunks (List[str]): 改进每个翻译块的专家建议。

    Returns:
        List[str]: 每个块的改进翻译。
    """

    system_message = f"你是语言专家，从事{source_lang}到{target_lang}的翻译和编辑工作。"

    improvement_prompt = """你的任务是仔细阅读原文本、初始翻译和语言学专家家建议，充分考虑语言学专家家建议，然后重新编辑初始翻译，输出最终翻译。

我会给你提供五段文本，分别由XML标记<PREV></PREV>、<TRANSLATE_THIS></TRANSLATE_THIS>、<NEXT></NEXT>、<TRANSLATION></TRANSLATION>、<EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS>包裹，<TRANSLATE_THIS></TRANSLATE_THIS>包裹的是要翻译的文本，<TRANSLATION></TRANSLATION>包裹的是翻译后的文本，<PREV></PREV>包裹的是原文的前文，<NEXT></NEXT>包裹的是原文的后文,<EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS>包裹的是语言学专家建议。

内容如下：
<PREV>{prev}</PREV>
<TRANSLATE_THIS>{translate_this}</TRANSLATE_THIS>
<NEXT>{next}</NEXT>


<TRANSLATION>{translation}</TRANSLATION>
<EXPERT_SUGGESTIONS>{expert_suggestions}</EXPERT_SUGGESTIONS>

只输出新翻译，没有其他内容。"""

    translation_2_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        # tagged_text = (
        #     "".join(source_text_chunks[0:i])
        #     + "<TRANSLATE_THIS>"
        #     + source_text_chunks[i]
        #     + "</TRANSLATE_THIS>"
        #     + "".join(source_text_chunks[i + 1 :])
        # )

        prompt = improvement_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            prev=source_text_chunks[i - 1] if i > 0 else "",
            next=source_text_chunks[i + 1] if i < len(source_text_chunks) - 1 else "",
            translate_this=source_text_chunks[i],
            translation=translation_1_chunks[i],
            expert_suggestions=reflection_chunks[i]
        )

        translation_2 = get_completion(prompt, system_message=system_message)
        translation_2_chunks.append(translation_2)

    return translation_2_chunks


def multichunk_translation(
    source_lang, target_lang, source_text_chunks, country: str = ""
):
    """
    Improves the translation of multiple text chunks based on the initial translation and reflection.

    Args:
        source_lang (str): The source language of the text chunks.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): The list of source text chunks to be translated.
        translation_1_chunks (List[str]): The list of initial translations for each source text chunk.
        reflection_chunks (List[str]): The list of reflections on the initial translations.
        country (str): Country specified for target language
    Returns:
        List[str]: The list of improved translations for each source text chunk.
    """

    translation_1_chunks = multichunk_initial_translation(
        source_lang, target_lang, source_text_chunks
    )

    reflection_chunks = multichunk_reflect_on_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        country,
    )

    translation_2_chunks = multichunk_improve_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        reflection_chunks,
    )

    return translation_2_chunks


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """
    根据令牌计数和令牌限制计算块大小。

    Args:
        token_count (int): The total number of tokens.
        token_limit (int): The maximum number of tokens allowed per chunk.

    Returns:
        int: The calculated chunk size.

    Description:
        此函数根据给定的令牌计数和令牌限制计算块大小。
        如果令牌计数小于或等于令牌限制，则函数将令牌计数作为块大小返回。
        否则，它会计算在令牌限制内容纳所有令牌所需的块数。
        块大小通过将令牌限制除以块数来确定。
        如果令牌计数除以令牌限制后还有剩余的令牌，
        块大小通过添加剩余的标记除以块的数量来调整。

    Example:
        >>> calculate_chunk_size(1000, 500)
        500
        >>> calculate_chunk_size(1530, 500)
        389
        >>> calculate_chunk_size(2242, 500)
        496
    """

    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


def translate(
    source_lang,
    target_lang,
    source_text,
    country,
    max_tokens=MAX_TOKENS_PER_CHUNK,
):
    """Translate the source_text from source_lang to target_lang."""

    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as single chunk")

        final_translation = one_chunk_translate_text(
            source_lang, target_lang, source_text, country
        )

        return final_translation

    else:
        ic("将文本翻译为多个块")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-3.5-turbo",
            chunk_size=token_size,
            chunk_overlap=0,
        )
        ic(text_splitter)
        source_text_chunks = text_splitter.split_text(source_text)
        ic("^".join(source_text_chunks))
        translation_2_chunks = multichunk_translation(
            source_lang, target_lang, source_text_chunks, country
        )

        return "".join(translation_2_chunks)
