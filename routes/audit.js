const express = require("express");
const { db } = require("../services/firebase");

const router = express.Router();

router.get("/", async (req, res, next) => {
  try {
    const caseId = req.query.caseId;
    let query = db().collection("auditEntries").orderBy("createdAt", "desc").limit(100);
    if (caseId) query = query.where("caseId", "==", caseId);

    const snapshot = await query.get();
    const entries = snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
    res.json({ success: true, entries });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
