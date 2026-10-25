// The example messages of the chat input bar (D-06; 11-UI-SPEC.md Copywriting, example messages): six short, generic
// Turkish banking messages written for this demo, in the spec's order. They are demo-authored and must never be copied
// from a split file: tests/test_demo_ui.py (test_examples_are_not_dataset_text, data marker) compares each with every
// message of the local data/splits files. Together they cover the Turkish letters the demo must render unchanged.
// This file holds exactly these six double-quoted strings (the test extracts them by that rule).

export const EXAMPLES: readonly string[] = [
  "Kartımı kaybettim, hemen dondurabilir misiniz?",
  "Hesabımdaki güncel bakiyeyi öğrenmek istiyorum.",
  "Yurt dışına para göndermenin ücreti ne kadar?",
  "İnternet şubesine giriş yapamıyorum, şifrem çalışmıyor.",
  "Otomatik ödeme talimatımı iptal etmek istiyorum.",
  "Merhaba, yardımınız için teşekkürler.",
];
