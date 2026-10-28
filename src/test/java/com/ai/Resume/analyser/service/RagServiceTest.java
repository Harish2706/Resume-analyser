package com.ai.Resume.analyser.service;

import com.ai.Resume.analyser.model.RagResponse;
import com.ai.Resume.analyser.model.RagSearchResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class RagServiceTest {

    private RagService ragService;

    @BeforeEach
    void setUp() {
        ragService = new RagService();
    }

    @Test
    void testChunkText() {
        String doc = "Section 1: Contact Information\nJohn Doe, john@example.com\n\n" +
                "Section 2: Technical Skills\nJava, Spring Boot, MySQL, REST APIs, Microservices\n\n" +
                "Section 3: Work Experience\nSoftware Engineer at Tech Corp. Developed scalable web services.";

        List<String> chunks = ragService.chunkText(doc, 100, 20);

        assertNotNull(chunks);
        assertFalse(chunks.isEmpty());
        assertTrue(chunks.stream().anyMatch(c -> c.contains("Technical Skills")));
    }

    @Test
    void testCosineSimilarityIdenticalVectors() {
        List<Float> v1 = Arrays.asList(0.6f, 0.8f);
        List<Float> v2 = Arrays.asList(0.6f, 0.8f);

        double similarity = ragService.cosineSimilarity(v1, v2);
        assertEquals(1.0, similarity, 0.001);
    }

    @Test
    void testCosineSimilarityOrthogonalVectors() {
        List<Float> v1 = Arrays.asList(1.0f, 0.0f);
        List<Float> v2 = Arrays.asList(0.0f, 1.0f);

        double similarity = ragService.cosineSimilarity(v1, v2);
        assertEquals(0.0, similarity, 0.001);
    }

    @Test
    void testLocalEmbeddingNormalization() {
        List<Float> embedding = ragService.generateLocalEmbedding("Java Spring Boot Developer");
        assertEquals(128, embedding.size());

        double norm = 0.0;
        for (float val : embedding) {
            norm += val * val;
        }
        assertEquals(1.0, Math.sqrt(norm), 0.001);
    }

    @Test
    void testSemanticSearchRanksRelevantChunkHighest() {
        String chunk1 = "Extensive experience with Python, Machine Learning, TensorFlow, and Pandas data pipelines.";
        String chunk2 = "Graphic design specialist skilled in Adobe Photoshop, Illustrator, and typography.";
        String chunk3 = "Bachelor of Science in Computer Science, Graduated 2023 with Honors.";

        List<String> chunks = Arrays.asList(chunk1, chunk2, chunk3);
        String query = "Python Machine Learning developer";

        List<RagSearchResult> results = ragService.semanticSearch(query, chunks, 2);

        assertEquals(2, results.size());
        assertEquals(0, results.get(0).getChunkIndex(), "Chunk 1 about Python ML should have highest similarity");
        assertTrue(results.get(0).getSimilarityScore() > results.get(1).getSimilarityScore());
    }

    @Test
    void testRetrieveContext() {
        String doc = "Professional Experience:\n" +
                "Built high-throughput distributed microservices in Java and Kubernetes.\n\n" +
                "Education:\n" +
                "University of Engineering, Degree in Computer Science.\n\n" +
                "Certifications:\n" +
                "AWS Certified Solutions Architect.";

        String context = ragService.retrieveContext("microservices Java", doc, 1);

        assertNotNull(context);
        assertTrue(context.contains("Java"));
        assertTrue(context.contains("microservices"));
    }

    @Test
    void testProcessRag() {
        String doc = "Skills:\nReact, TypeScript, Next.js, Redux, CSS Modules.\n\n" +
                "Projects:\nCreated an e-commerce platform with React and Stripe.";

        RagResponse response = ragService.processRag("React Frontend", doc, 2, false);

        assertNotNull(response);
        assertEquals("React Frontend", response.getQuery());
        assertFalse(response.getResults().isEmpty());
        assertNotNull(response.getRetrievedContext());
    }
}

