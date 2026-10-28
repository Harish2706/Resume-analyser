package com.ai.Resume.analyser.service;

import com.ai.Resume.analyser.model.RagResponse;
import com.ai.Resume.analyser.model.RagSearchResult;
import com.google.genai.Client;
import com.google.genai.types.Content;
import com.google.genai.types.ContentEmbedding;
import com.google.genai.types.EmbedContentResponse;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Part;
import org.apache.tika.Tika;
import org.apache.tika.exception.TikaException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Service
public class RagService {

    @Value("${genKey:}")
    private String genKey;

    public List<String> chunkText(String text, int maxChunkSize, int overlap) {
        List<String> chunks = new ArrayList<>();
        if (text == null || text.trim().isEmpty()) {
            return chunks;
        }

        String normalized = text.replace("\r\n", "\n").replace("\r", "\n").trim();
        String[] paragraphs = normalized.split("\n\\s*\n+");

        StringBuilder currentChunk = new StringBuilder();

        for (String para : paragraphs) {
            String trimmedPara = para.trim();
            if (trimmedPara.isEmpty()) {
                continue;
            }

            if (trimmedPara.length() > maxChunkSize) {
                if (currentChunk.length() > 0) {
                    chunks.add(currentChunk.toString().trim());
                    currentChunk.setLength(0);
                }

                int start = 0;
                while (start < trimmedPara.length()) {
                    int end = Math.min(start + maxChunkSize, trimmedPara.length());
                    if (end < trimmedPara.length()) {
                        int lastSpace = trimmedPara.lastIndexOf(' ', end);
                        if (lastSpace > start + (maxChunkSize / 2)) {
                            end = lastSpace;
                        }
                    }
                    String sub = trimmedPara.substring(start, end).trim();
                    if (!sub.isEmpty()) {
                        chunks.add(sub);
                    }
                    if (end >= trimmedPara.length()) {
                        break;
                    }
                    start = Math.max(end - overlap, start + 1);
                }
            } else {
                if (currentChunk.length() + trimmedPara.length() + 2 <= maxChunkSize) {
                    if (currentChunk.length() > 0) {
                        currentChunk.append("\n\n");
                    }
                    currentChunk.append(trimmedPara);
                } else {
                    if (currentChunk.length() > 0) {
                        chunks.add(currentChunk.toString().trim());
                    }
                    currentChunk = new StringBuilder(trimmedPara);
                }
            }
        }

        if (currentChunk.length() > 0) {
            chunks.add(currentChunk.toString().trim());
        }

        return chunks;
    }

    public List<Float> getEmbedding(String text) {
        if (text == null || text.trim().isEmpty()) {
            return generateLocalEmbedding("");
        }

        if (genKey != null && !genKey.trim().isEmpty()) {
            try {
                Client client = Client.builder().apiKey(genKey).build();
                EmbedContentResponse response = client.models.embedContent("text-embedding-004", text, null);
                if (response != null && response.embeddings().isPresent()) {
                    List<ContentEmbedding> embeddings = response.embeddings().get();
                    if (!embeddings.isEmpty() && embeddings.get(0).values().isPresent()) {
                        return embeddings.get(0).values().get();
                    }
                }
            } catch (Exception e) {
                System.err.println("Google GenAI embedding call failed, using local vector fallback: " + e.getMessage());
            }
        }

        return generateLocalEmbedding(text);
    }

    public List<Float> generateLocalEmbedding(String text) {
        int dimension = 128;
        float[] vector = new float[dimension];

        if (text != null && !text.trim().isEmpty()) {
            String[] tokens = text.toLowerCase().replaceAll("[^a-z0-9\\s]", " ").split("\\s+");
            for (String token : tokens) {
                if (token.length() >= 2) {
                    int hash = Math.abs(token.hashCode()) % dimension;
                    vector[hash] += 1.0f;
                }
            }
        }

        double norm = 0.0;
        for (float val : vector) {
            norm += val * val;
        }
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < dimension; i++) {
                vector[i] = (float) (vector[i] / norm);
            }
        }

        List<Float> result = new ArrayList<>(dimension);
        for (float val : vector) {
            result.add(val);
        }
        return result;
    }

    public double cosineSimilarity(List<Float> vectorA, List<Float> vectorB) {
        if (vectorA == null || vectorB == null || vectorA.isEmpty() || vectorB.isEmpty()) {
            return 0.0;
        }

        int size = Math.min(vectorA.size(), vectorB.size());
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < size; i++) {
            float a = vectorA.get(i);
            float b = vectorB.get(i);
            dotProduct += a * b;
            normA += a * a;
            normB += b * b;
        }

        double denominator = Math.sqrt(normA) * Math.sqrt(normB);
        if (denominator == 0.0) {
            return 0.0;
        }

        return dotProduct / denominator;
    }

    public List<RagSearchResult> semanticSearch(String query, List<String> chunks, int topK) {
        if (chunks == null || chunks.isEmpty()) {
            return Collections.emptyList();
        }

        List<Float> queryEmbedding = getEmbedding(query);
        List<RagSearchResult> rankedResults = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            String chunk = chunks.get(i);
            List<Float> chunkEmbedding = getEmbedding(chunk);
            double similarity = cosineSimilarity(queryEmbedding, chunkEmbedding);
            rankedResults.add(new RagSearchResult(i, chunk, similarity));
        }

        rankedResults.sort((a, b) -> Double.compare(b.getSimilarityScore(), a.getSimilarityScore()));

        int limit = Math.min(topK > 0 ? topK : 3, rankedResults.size());
        return new ArrayList<>(rankedResults.subList(0, limit));
    }

    public String retrieveContext(String query, String documentText, int topK) {
        List<String> chunks = chunkText(documentText, 400, 50);
        List<RagSearchResult> topResults = semanticSearch(query, chunks, topK);

        StringBuilder contextBuilder = new StringBuilder();
        for (int i = 0; i < topResults.size(); i++) {
            RagSearchResult result = topResults.get(i);
            contextBuilder.append(String.format("[Source Chunk %d | Similarity: %.2f]\n%s\n\n",
                    result.getChunkIndex() + 1, result.getSimilarityScore(), result.getChunkText()));
        }
        return contextBuilder.toString().trim();
    }

    public String generateAnswerFromContext(String query, String retrievedContext) {
        if (retrievedContext == null || retrievedContext.trim().isEmpty()) {
            return "No relevant context found in the document for the query: " + query;
        }

        if (genKey != null && !genKey.trim().isEmpty()) {
            try {
                Client client = Client.builder().apiKey(genKey).build();
                String promptText = "You are an AI assistant analyzing a resume/document.\n" +
                        "Use ONLY the following retrieved context to answer the user's question. " +
                        "If the answer cannot be found in the context, clearly state that the document does not contain this information.\n\n" +
                        "Retrieved Context:\n" + retrievedContext + "\n\n" +
                        "Question: " + query + "\n\n" +
                        "Answer:";

                Content content = Content.builder().parts(Part.fromText(promptText)).build();
                GenerateContentConfig config = GenerateContentConfig.builder()
                        .temperature(0.2f)
                        .build();

                GenerateContentResponse response = client.models.generateContent("gemini-3.5-flash-lite", content, config);
                if (response != null && response.text() != null) {
                    return response.text().trim();
                }
            } catch (Exception e) {
                System.err.println("Gemini answer generation failed: " + e.getMessage());
            }
        }

        return "Retrieved context successfully matching the query. (Gemini generation unavailable without an active API key).";
    }

    public RagResponse processRag(String query, String documentText, int topK, boolean generateAnswer) {
        List<String> chunks = chunkText(documentText, 400, 50);
        List<RagSearchResult> results = semanticSearch(query, chunks, topK);

        StringBuilder contextBuilder = new StringBuilder();
        for (RagSearchResult result : results) {
            contextBuilder.append(String.format("[Source Chunk %d | Similarity: %.2f]\n%s\n\n",
                    result.getChunkIndex() + 1, result.getSimilarityScore(), result.getChunkText()));
        }
        String retrievedContext = contextBuilder.toString().trim();

        String answer = null;
        if (generateAnswer) {
            answer = generateAnswerFromContext(query, retrievedContext);
        }

        return new RagResponse(query, retrievedContext, results, answer);
    }

    public String extractTextFromFile(MultipartFile file) throws IOException, TikaException {
        Tika tika = new Tika();
        try (ByteArrayInputStream is = new ByteArrayInputStream(file.getBytes())) {
            return tika.parseToString(is);
        }
    }
}
