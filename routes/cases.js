const express = require("express");
const { db, timestamp } = require("../services/firebase");
const { createCaseId } = require("../utils/caseId");
const { addAuditEntry } = require("../services/auditService");

const router = express.Router();

router.post("/", async (req, res, next) => {
  try {
    const caseId = createCaseId();
    const now = new Date().toISOString();

    const caseData = {
      caseId,
      status: "Processing",
      riskScore: null,
      riskLabel: null,
      applicant: {},
      document: {},
      verification: {
        ocr: { status: "Pending" },
        validation: { status: "Pending" },
        forensics: { status: "Pending" },
        biometric: { status: "Pending" },
        risk: { status: "Pending" },
      },
      createdAt: timestamp(),
      updatedAt: timestamp(),
      createdAtIso: now,
    };

    await db().collection("cases").doc(caseId).set(caseData);
    await addAuditEntry({
      caseId,
      action: "Verification Case Created",
      module: "Case",
      status: "Created",
    });

    res.status(201).json({ success: true, case: { ...caseData, createdAt: now, updatedAt: now } });
  } catch (error) {
    next(error);
  }
});

router.get("/", async (req, res, next) => {
  try {
    const limit = Math.min(Number(req.query.limit || 50), 100);
    const snapshot = await db()
      .collection("cases")
      .orderBy("createdAt", "desc")
      .limit(limit)
      .get();

    const cases = snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
    res.json({ success: true, cases });
  } catch (error) {
    next(error);
  }
});

router.get("/:caseId", async (req, res, next) => {
  try {
    const doc = await db().collection("cases").doc(req.params.caseId).get();
    if (!doc.exists) return res.status(404).json({ success: false, message: "Case not found" });
    res.json({ success: true, case: { id: doc.id, ...doc.data() } });
  } catch (error) {
    next(error);
  }
});

router.patch("/:caseId", async (req, res, next) => {
  try {
    const allowed = ["status", "riskScore", "riskLabel", "applicant", "document", "verification"];
    const update = {};
    for (const key of allowed) {
      if (req.body[key] !== undefined) update[key] = req.body[key];
    }
    update.updatedAt = timestamp();

    await db().collection("cases").doc(req.params.caseId).update(update);
    const updated = await db().collection("cases").doc(req.params.caseId).get();
    res.json({ success: true, case: { id: updated.id, ...updated.data() } });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
