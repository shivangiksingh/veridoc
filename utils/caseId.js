function createCaseId() {
  const now = new Date();
  const year = now.getFullYear();
  const random = Math.floor(100000 + Math.random() * 900000);
  return `VRD-${year}-${random}`;
}

module.exports = { createCaseId };
