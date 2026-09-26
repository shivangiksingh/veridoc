const express = require("express");
const upload = require("../middleware/upload");
const { addAuditEntry } = require("../services/auditService");
const { db, timestamp } = require("../services/firebase");

const router = express.Router();

async function callAi(path, formData) {
  const base = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";
  const response = await fetch(`${base}${path}`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`AI service error ${response.status}: ${text}`);
  }
  return response.json();
}

router.post("/upload", upload.single("document"), async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ success: false, message: "Document is required" });

    const { caseId } = req.body;
    if (!caseId) return res.status(400).json({ success: false, message: "caseId is required" });

    await db().collection("cases").doc(caseId).update({
      "document.fileName": req.file.originalname,
      "document.mimeType": req.file.mimetype,
      "document.size": req.file.size,
      "document.uploadedAt": timestamp(),
      updatedAt: timestamp(),
    });

    await addAuditEntry({
      caseId,
      action: "Identity Document Uploaded",
      module: "Upload",
      status: "Completed",
      details: { fileName: req.file.originalname, mimeType: req.file.mimetype, size: req.file.size },
    });

    res.status(201).json({
      success: true,
      file: {
        name: req.file.originalname,
        mimeType: req.file.mimetype,
        size: req.file.size,
      },
    });
  } catch (error) {
    next(error);
  }
});

router.post("/ocr", upload.single("document"), async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ success: false, message: "Document is required" });
    const form = new FormData();
    form.append("document", new Blob([req.file.buffer], { type: req.file.mimetype }), req.file.originalname);
    const result = await callAi("/ocr", form);
    res.json({ success: true, result });
  } catch (error) {
    next(error);
  }
});

router.post("/validate", async (req, res, next) => {
  try {
    const result = await fetch(`${process.env.AI_SERVICE_URL || "http://127.0.0.1:8000"}/validate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    const data = await result.json();
    res.status(result.status).json({ success: result.ok, result: data });
  } catch (error) {
    next(error);
  }
});

router.post("/forensics", upload.single("document"), async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ success: false, message: "Document is required" });
    const form = new FormData();
    form.append("document", new Blob([req.file.buffer], { type: req.file.mimetype }), req.file.originalname);
    const result = await callAi("/forensics", form);
    res.json({ success: true, result });
  } catch (error) {
    next(error);
  }
});

router.post("/biometric", upload.fields([
  { name: "documentFace", maxCount: 1 },
  { name: "livePhoto", maxCount: 1 },
]), async (req, res, next) => {
  try {
    if (!req.files?.documentFace?.[0] || !req.files?.livePhoto?.[0]) {
      return res.status(400).json({ success: false, message: "documentFace and livePhoto are required" });
    }

    const form = new FormData();
    const documentFace = req.files.documentFace[0];
    const livePhoto = req.files.livePhoto[0];
    form.append("documentFace", new Blob([documentFace.buffer], { type: documentFace.mimetype }), documentFace.originalname);
    form.append("livePhoto", new Blob([livePhoto.buffer], { type: livePhoto.mimetype }), livePhoto.originalname);

    const result = await callAi("/biometric", form);
    res.json({ success: true, result });
  } catch (error) {
    next(error);
  }
});

router.post("/risk", async (req, res, next) => {
  try {
    const response = await fetch(`${process.env.AI_SERVICE_URL || "http://127.0.0.1:8000"}/risk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    const data = await response.json();
    res.status(response.status).json({ success: response.ok, result: data });
  } catch (error) {
    next(error);
  }
});

router.post("/:caseId/audit", async (req, res, next) => {
  try {
    const entry = await addAuditEntry({
      caseId: req.params.caseId,
      action: req.body.action || "Verification Event",
      module: req.body.module || "Verification",
      status: req.body.status || "Completed",
      details: req.body.details || {},
      user: req.body.user || "Admin User",
    });
    res.status(201).json({ success: true, entry });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
