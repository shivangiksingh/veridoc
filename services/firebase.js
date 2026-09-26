const admin = require("firebase-admin");

let app;

function initFirebase() {
  if (app) return app;

  if (admin.apps.length) {
    app = admin.app();
    return app;
  }

  const hasExplicitCredentials =
    process.env.FIREBASE_PROJECT_ID &&
    process.env.FIREBASE_CLIENT_EMAIL &&
    process.env.FIREBASE_PRIVATE_KEY;

  if (hasExplicitCredentials) {
    app = admin.initializeApp({
      credential: admin.credential.cert({
        projectId: process.env.FIREBASE_PROJECT_ID,
        clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
        privateKey: process.env.FIREBASE_PRIVATE_KEY.replace(/\\n/g, "\n"),
      }),
    });
  } else {
    app = admin.initializeApp({
      credential: admin.credential.applicationDefault(),
    });
  }

  return app;
}

function db() {
  initFirebase();
  return admin.firestore();
}

function timestamp() {
  initFirebase();
  return admin.firestore.FieldValue.serverTimestamp();
}

module.exports = { admin, initFirebase, db, timestamp };
