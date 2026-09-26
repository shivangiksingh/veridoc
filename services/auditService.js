const { db, timestamp } = require("./firebase");
const { sha256 } = require("../utils/hash");

async function getPreviousHash(caseId) {
  const snap = await db()
    .collection("auditEntries")
    .where("caseId", "==", caseId)
    .orderBy("createdAt", "desc")
    .limit(1)
    .get();

  if (snap.empty) return "GENESIS";
  return snap.docs[0].data().hash || "GENESIS";
}

async function addAuditEntry({ caseId, action, module, status, details = {}, user = "Admin User" }) {
  const previousHash = await getPreviousHash(caseId);
  const payload = {
    caseId,
    action,
    module,
    status,
    details,
    user,
    previousHash,
    createdAtIso: new Date().toISOString(),
  };
  const hash = sha256(JSON.stringify(payload));

  const doc = {
    ...payload,
    hash,
    createdAt: timestamp(),
  };

  const ref = await db().collection("auditEntries").add(doc);
  return { id: ref.id, ...doc, createdAt: payload.createdAtIso };
}

module.exports = { addAuditEntry };
