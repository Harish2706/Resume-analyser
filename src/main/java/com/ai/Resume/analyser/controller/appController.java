package com.ai.Resume.analyser.controller;


import com.ai.Resume.analyser.model.RagResponse;
import com.ai.Resume.analyser.model.RagSearchRequest;
import com.ai.Resume.analyser.service.RagService;
import com.ai.Resume.analyser.service.appService;
import org.apache.tika.exception.TikaException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("resumeAnalyserCore/service/v1")
@CrossOrigin( origins = "http://localhost:5173" , allowCredentials = "true", allowedHeaders = "*" , methods = {RequestMethod.GET,RequestMethod.POST,RequestMethod.DELETE,RequestMethod.PUT,RequestMethod.OPTIONS,RequestMethod.HEAD})
public class appController {

    @Autowired
    private appService appServices;

    @Autowired
    private RagService ragService;

    @PostMapping("/extract")
    public ResponseEntity<?> extract(@RequestParam String roles, @RequestParam MultipartFile file) throws TikaException, IOException, InterruptedException {
        return appServices.extract(roles,file);
    }

    @PostMapping("/rag/search")
    public ResponseEntity<?> ragSearch(@RequestBody RagSearchRequest req) {
        if (req == null || req.getQuery() == null || req.getQuery().trim().isEmpty()) {
            return ResponseEntity.badRequest().body("Query cannot be empty");
        }
        if (req.getDocumentText() == null || req.getDocumentText().trim().isEmpty()) {
            return ResponseEntity.badRequest().body("Document text cannot be empty");
        }
        int topK = req.getTopK() > 0 ? req.getTopK() : 3;
        RagResponse response = ragService.processRag(req.getQuery(), req.getDocumentText(), topK, false);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/rag/ask")
    public ResponseEntity<?> ragAsk(@RequestBody RagSearchRequest req) {
        if (req == null || req.getQuery() == null || req.getQuery().trim().isEmpty()) {
            return ResponseEntity.badRequest().body("Query cannot be empty");
        }
        if (req.getDocumentText() == null || req.getDocumentText().trim().isEmpty()) {
            return ResponseEntity.badRequest().body("Document text cannot be empty");
        }
        int topK = req.getTopK() > 0 ? req.getTopK() : 3;
        RagResponse response = ragService.processRag(req.getQuery(), req.getDocumentText(), topK, true);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/rag/search-file")
    public ResponseEntity<?> ragSearchFile(@RequestParam("file") MultipartFile file,
                                          @RequestParam("query") String query,
                                          @RequestParam(value = "topK", defaultValue = "3") int topK) throws IOException, TikaException {
        if (file == null || file.isEmpty()) {
            return ResponseEntity.badRequest().body("File cannot be empty");
        }
        if (query == null || query.trim().isEmpty()) {
            return ResponseEntity.badRequest().body("Query cannot be empty");
        }
        String text = ragService.extractTextFromFile(file);
        RagResponse response = ragService.processRag(query, text, topK, false);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/rag/ask-file")
    public ResponseEntity<?> ragAskFile(@RequestParam("file") MultipartFile file,
                                       @RequestParam("query") String query,
                                       @RequestParam(value = "topK", defaultValue = "3") int topK) throws IOException, TikaException {
        if (file == null || file.isEmpty()) {
            return ResponseEntity.badRequest().body("File cannot be empty");
        }
        if (query == null || query.trim().isEmpty()) {
            return ResponseEntity.badRequest().body("Query cannot be empty");
        }
        String text = ragService.extractTextFromFile(file);
        RagResponse response = ragService.processRag(query, text, topK, true);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/lastReport")
    public ResponseEntity<?> lastReport(){
        return appServices.lastReport();
    }

    @PostMapping("/logout")
    public ResponseEntity<?> logout(){
        return  appServices.logout();
    }

    @PostMapping("/deleteAccount")
    public ResponseEntity<?> deleteAccount(){
        return  appServices.deleteAccount();
    }

    @PostMapping("/isValid")
    public ResponseEntity<?> tokenValidation(){
        return appServices.tokenValidation();
    }

}
