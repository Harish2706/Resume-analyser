
package com.ai.Resume.analyser.service;

import com.ai.Resume.analyser.model.RagChunkEmbedding;
import com.ai.Resume.analyser.model.RagResponse;
import com.ai.Resume.analyser.model.RagSearchResult;
import com.ai.Resume.analyser.repository.RagChunkEmbeddingRepository;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.genai.Client;
import com.google.genai.types.Content;
import com.google.genai.types.ContentEmbedding;
import com.google.genai.types.EmbedContentConfig;
import com.google.genai.types.EmbedContentResponse;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Part;
import com.google.genai.types.TaskType;
import org.apache.tika.Tika;
import org.apache.tika.exception.TikaException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;

@Service
public class RagService {

    @Value("${genKey:}")
    private String genKey;

    @Value("${genai.embedding-model:gemini-embedding-001}")
    private String embeddingModel;

    @Value("${genai.generation-model:gemini-3.5-flash-lite}")
    private String generationModel;

    private final RagChunkEmbeddingRepository embeddingRepository;
    private final ObjectMapper objectMapper;

    public RagService(
            RagChunkEmbeddingRepository embeddingRepository,
            ObjectMapper objectMapper) {
        this.embeddingRepository = embeddingRepository;
        this.objectMapper = objectMapper;
    }

   
    public List<String> chunkText(
            String text, int maxChunkSize, int overlap) {

        if (text == null || text.isBlank()) {
            return Collections.emptyList();
        }

        if (maxChunkSize <= 0 || overlap < 0
                || overlap >= maxChunkSize) {
            throw new IllegalArgumentException(
                    "Invalid chunk size or overlap");
        }

        List<String> chunks = new ArrayList<>();

        String normalized = text.replace("\r\n", "\n")
                .replace("\r", "\n").trim();

        String[] paragraphs = normalized.split("\n\\s*\n+");
        StringBuilder current = new StringBuilder();

        for (String paragraph : paragraphs) {
            String p = paragraph.trim();

            if (p.isEmpty()) {
                continue;
            }

            if (p.length() > maxChunkSize) {
                if (current.length() > 0) {
                    chunks.add(current.toString().trim());
                    current.setLength(0);
                }

                int start = 0;

                while (start < p.length()) {
                    int end = Math.min(
                            start + maxChunkSize, p.length());

                    if (end < p.length()) {
                        int lastSpace = p.lastIndexOf(' ', end);

                        if (lastSpace > start + maxChunkSize / 2) {
                            end = lastSpace;
                        }
                    }

                    String part = p.substring(start, end).trim();

                    if (!part.isEmpty()) {
                        chunks.add(part);
                    }

                    if (end >= p.length()) {
                        break;
                    }

                    start = Math.max(end - overlap, start + 1);
                }
            } else {
                int separator = current.length() == 0 ? 0 : 2;

                if (current.length() + separator + p.length()
                        <= maxChunkSize) {
                    if (current.length() > 0) {
                        current.append("\n\n");
                    }
                    current.append(p);
                } else {
                    if (current.length() > 0) {
                        chunks.add(current.toString().trim());
                    }
                    current = new StringBuilder(p);
                }
            }
        }

        if (current.length() > 0) {
            chunks.add(current.toString().trim());
        }

        return chunks;
    }

   
    public List<Float> getEmbedding(String text) {
        return generateGoogleEmbedding(text, TaskType.RETRIEVAL_QUERY);
    }

    
    private List<Float> getDocumentEmbedding(String chunk) {

        String cacheKey = sha256(embeddingModel + ":" + chunk);

        return embeddingRepository.findByChunkHash(cacheKey)
                .map(saved -> {
                    try {
                        return objectMapper.readValue(
                                saved.getEmbeddingJson(),
                                new TypeReference<List<Float>>() {});
                    } catch (IOException e) {
                        throw new IllegalStateException(
                                "Failed to read saved embedding", e);
                    }
                })
                .orElseGet(() -> {
                    List<Float> vector = generateGoogleEmbedding(
                            chunk, TaskType.RETRIEVAL_DOCUMENT);

                    try {
                        RagChunkEmbedding saved = new RagChunkEmbedding();
                        saved.setChunkHash(cacheKey);
                        saved.setChunkText(chunk);
                        saved.setEmbeddingModel(embeddingModel);
                        saved.setEmbeddingJson(
                                objectMapper.writeValueAsString(vector));

                        embeddingRepository.save(saved);

                        return vector;
                    } catch (IOException e) {
                        throw new IllegalStateException(
                                "Failed to save embedding JSON", e);
                    }
                });
    }


    private List<Float> generateGoogleEmbedding(
            String text, TaskType taskType) {

        if (text == null || text.isBlank()) {
            return Collections.emptyList();
        }

        if (genKey == null || genKey.isBlank()) {
            throw new IllegalStateException(
                    "Configure the Gemini API key before using RAG");
        }

        try {
            Client client = Client.builder()
                    .apiKey(genKey)
                    .build();

            EmbedContentConfig config = EmbedContentConfig.builder()
                    .taskType(taskType)
                    .outputDimensionality(768)
                    .build();

            EmbedContentResponse response =
                    client.models.embedContent(
                            embeddingModel, text, config);

            if (response != null
                    && response.embeddings().isPresent()) {

                List<ContentEmbedding> embeddings =
                        response.embeddings().get();

                if (!embeddings.isEmpty()
                        && embeddings.get(0).values().isPresent()) {
                    return embeddings.get(0).values().get();
                }
            }

            throw new IllegalStateException(
                    "Embedding API returned no vector");

        } catch (Exception e) {
            throw new IllegalStateException(
                    "Embedding generation failed: " + e.getMessage(), e);
        }
    }

 
    public double cosineSimilarity(
            List<Float> a, List<Float> b) {

        if (a == null || b == null || a.isEmpty()
                || a.size() != b.size()) {
            return 0.0;
        }

        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < a.size(); i++) {
            double x = a.get(i);
            double y = b.get(i);

            dot += x * y;
            normA += x * x;
            normB += y * y;
        }

        double denominator = Math.sqrt(normA) * Math.sqrt(normB);

        return denominator == 0.0 ? 0.0 : dot / denominator;
    }

  
    public List<RagSearchResult> semanticSearch(
            String query, List<String> chunks, int topK) {

        if (chunks == null || chunks.isEmpty()) {
            return Collections.emptyList();
        }

        List<Float> queryVector = getEmbedding(query);
        List<RagSearchResult> results = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            String chunk = chunks.get(i);

            // Uses MySQL cache on repeated searches.
            List<Float> chunkVector = getDocumentEmbedding(chunk);

            double score = cosineSimilarity(queryVector, chunkVector);

            results.add(new RagSearchResult(i, chunk, score));
        }

        results.sort((a, b) -> Double.compare(
                b.getSimilarityScore(), a.getSimilarityScore()));

        int limit = Math.min(topK > 0 ? topK : 3, results.size());

        return new ArrayList<>(results.subList(0, limit));
    }

    private String buildContext(List<RagSearchResult> results) {
        StringBuilder context = new StringBuilder();

        for (RagSearchResult result : results) {
            context.append(String.format(
                    "[Source Chunk %d | Similarity: %.4f]%n%s%n%n",
                    result.getChunkIndex() + 1,
                    result.getSimilarityScore(),
                    result.getChunkText()));
        }

        return context.toString().trim();
    }

    public String retrieveContext(
            String query, String documentText, int topK) {

        List<String> chunks = chunkText(documentText, 400, 50);

        return buildContext(semanticSearch(query, chunks, topK));
    }


    public String generateAnswerFromContext(
            String query, String retrievedContext) {

        if (retrievedContext == null || retrievedContext.isBlank()) {
            return "No relevant context found for the query: " + query;
        }

        if (genKey == null || genKey.isBlank()) {
            throw new IllegalStateException(
                    "Configure the Gemini API key before generation");
        }

        try {
            Client client = Client.builder()
                    .apiKey(genKey)
                    .build();

            String prompt = """
                    You are an assistant analyzing a resume or document.
                    Answer using only the retrieved context.
                    If the answer is not supported by the context,
                    clearly state that there is insufficient information.
                    Do not invent qualifications, skills, or experience.

                    Retrieved context:
                    %s

                    Question:
                    %s

                    Answer:
                    """.formatted(retrievedContext, query);

            Content content = Content.builder()
                    .parts(Part.fromText(prompt))
                    .build();

            GenerateContentConfig config =
                    GenerateContentConfig.builder()
                            .temperature(0.2f)
                            .build();

            GenerateContentResponse response =
                    client.models.generateContent(
                            generationModel, content, config);

            if (response != null && response.text() != null) {
                return response.text().trim();
            }

            throw new IllegalStateException(
                    "Gemini returned an empty answer");

        } catch (Exception e) {
            throw new IllegalStateException(
                    "Answer generation failed: " + e.getMessage(), e);
        }
    }

    public RagResponse processRag(
            String query,
            String documentText,
            int topK,
            boolean generateAnswer) {

        List<String> chunks = chunkText(documentText, 400, 50);
        List<RagSearchResult> results =
                semanticSearch(query, chunks, topK);

        String context = buildContext(results);

        String answer = generateAnswer
                ? generateAnswerFromContext(query, context)
                : null;

        return new RagResponse(query, context, results, answer);
    }

    public String extractTextFromFile(MultipartFile file)
            throws IOException, TikaException {

        Tika tika = new Tika();

        try (ByteArrayInputStream input =
                     new ByteArrayInputStream(file.getBytes())) {
            return tika.parseToString(input);
        }
    }

    private String sha256(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(
                    value.getBytes(StandardCharsets.UTF_8));

            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(
                    "SHA-256 is unavailable", e);
        }
    }
}
