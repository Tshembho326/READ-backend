from django.test import TestCase
import numpy as np

def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-1):
    # Create score and traceback matrices
    n = len(seq1) + 1
    m = len(seq2) + 1
    score_matrix = np.zeros((n, m))
    traceback_matrix = np.zeros((n, m), dtype=int)

    # Initialize score matrix
    for i in range(n):
        score_matrix[i][0] = i * gap
    for j in range(m):
        score_matrix[0][j] = j * gap

    # Fill matrices
    for i in range(1, n):
        for j in range(1, m):
            match_score = score_matrix[i - 1][j - 1] + (match if seq1[i - 1] == seq2[j - 1] else mismatch)
            delete_score = score_matrix[i - 1][j] + gap
            insert_score = score_matrix[i][j - 1] + gap

            score_matrix[i][j] = max(match_score, delete_score, insert_score)

            if score_matrix[i][j] == match_score:
                traceback_matrix[i][j] = 1  # Diagonal
            elif score_matrix[i][j] == delete_score:
                traceback_matrix[i][j] = 2  # Up
            else:
                traceback_matrix[i][j] = 3  # Left

    # Traceback
    align1, align2 = [], []
    i, j = n - 1, m - 1

    while i > 0 or j > 0:
        if traceback_matrix[i][j] == 1:
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == 2:
            align1.append(seq1[i - 1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j - 1])
            j -= 1

    # Add remaining gaps
    while i > 0:
        align1.append(seq1[i - 1])
        align2.append('-')
        i -= 1

    while j > 0:
        align1.append('-')
        align2.append(seq2[j - 1])
        j -= 1

    align1.reverse()
    align2.reverse()

    # Return aligned sequences
    return ''.join(align1), ''.join(align2)


# Example usage
transcription_phonemes = "d ə k ɪ ɡ b r aʊ n f ɔ k s".split()
story_phonemes = "ð ə k w ˈɪ k b ɹ ˈaʊ n f ˈɒ k s".split()

aligned_transcription, aligned_story = needleman_wunsch(transcription_phonemes, story_phonemes)

# Print results
print("Aligned Transcription: ", aligned_transcription)
print("Aligned Story Phonemes: ", aligned_story)


def compare_aligned_sequences(seq1, seq2):
    results = []
    for s1, s2 in zip(seq1, seq2):
        if s1 == s2:
            results.append(f"Matched: {s1}")
        elif s1 == '-' or s2 == '-':
            results.append(f"Gap: {s1} vs {s2}")
        else:
            results.append(f"Mismatch: expected '{s2}', but got '{s1}'")
    return results

# Example usage
comparison_results = compare_aligned_sequences(aligned_transcription, aligned_story)
for result in comparison_results:
    print(result)
