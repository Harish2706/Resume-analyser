package com.ai.Resume.analyser.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class RagSearchResult {
    private int chunkIndex;
    private String chunkText;
    private double similarityScore;
}

