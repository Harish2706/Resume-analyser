package com.ai.Resume.analyser.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class RagResponse {
    private String query;
    private String retrievedContext;
    private List<RagSearchResult> results;
    private String generatedAnswer;
}

