#include "whisper.h"
#include "common-whisper.h"

#include <cstdio>
#include <string>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

int main() {
    std::string whisper_model_path = WHISPER_MODEL_PATH;
    std::string sample_path        = SAMPLE_PATH;

    // Load the sample audio file
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    assert(read_audio_data(sample_path.c_str(), pcmf32, pcmf32s, false));

    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context * wctx = whisper_init_from_file_with_params(
            whisper_model_path.c_str(),
            cparams);
    assert(wctx != nullptr);

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);

    assert(whisper_full_parallel(wctx, wparams, pcmf32.data(), pcmf32.size(), 1) == 0);

    // Collect the full transcription across all segments
    const int n_segments = whisper_full_n_segments(wctx);
    assert(n_segments > 0);

    std::string transcription;
    for (int i = 0; i < n_segments; ++i) {
        transcription += whisper_full_get_segment_text(wctx, i);
    }

    printf("Transcription:\n%s\n", transcription.c_str());

    // Verify the transcription matches the expected JFK quote
    const char * expected = " And so my fellow Americans, ask not what your country can do for you,"
                            " ask what you can do for your country.";
    assert(transcription == expected);

    whisper_free(wctx);

    return 0;
}
