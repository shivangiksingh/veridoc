const crypto = require("crypto");

function sha256(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function stablePayload(payload) {
  return JSON.stringify(payload, Object.keys(payload).sort());
}

module.exports = { sha256, stablePayload };
