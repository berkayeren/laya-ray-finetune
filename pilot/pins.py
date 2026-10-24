"""Every pinned revision and digest the pilot trusts (dataset, raw folds, model weights, OOD tag subset).

Values were verified against the Hugging Face API and by hashing the downloaded files. Change a value here only
together with a deliberate refreeze of data/splits.json.
"""

DATASET_REPO = "uoe-nlp/multi3-nlu"
DATASET_REVISION = "cde8e2537d7ac06212ef4b25df2cf3084f26ec0f"
DATASET_LICENSE = "cc-by-4.0"
DATASET_PATTERNS = ["turkish/banking/*", "turkish/hotels/*"]

# sha256 of every raw fold file, relative to the snapshot root
RAW_SHA256 = {
    "turkish/banking/fold0.json": "688643c9f8e48c0f7e34f3ffeb3a1e87fe4bea6d2350787a6f3fa463eff9c4d7",
    "turkish/banking/fold1.json": "935501f6098a7ad394b3f560fe0c30e81e0640ce2151e832c2e2bad31bfced8f",
    "turkish/banking/fold2.json": "c3bb0547b74a918340bd2507705db5d506aa3b1a774940f5a466332d02648579",
    "turkish/banking/fold3.json": "0526c088576a6a7942bfdc49f3d79391c98dd59894e6b84eb5fd57237d494570",
    "turkish/banking/fold4.json": "44621778cd1e24e0f790df7a1c5d03e4405380e31c25bc482dc2f9503fc7e976",
    "turkish/banking/fold5.json": "26d7a3de01c03ca115c1a009b026ed93a0acf9d031a80c8a4c38b8b1abf8d8ed",
    "turkish/banking/fold6.json": "a568617d3a1eb0e4c46926a2286ff46ccb60348aa0106c03aba20f01547ca9d5",
    "turkish/banking/fold7.json": "8ab40d0aab305db0003301b09b941d63a838e408d45738e3d1ebe1379539bdd3",
    "turkish/banking/fold8.json": "aae678824c2e8ba70e18be3565a70d1ed7d575844e1084a8858acfc72b74cc58",
    "turkish/banking/fold9.json": "9810ab4ef9dc4c53fdbe8715073871944b22ed6ce9a32c0d26a27a8c787a4757",
    "turkish/banking/fold10.json": "cfcdb52e1de912d5a9c89b699229e7ec2ae321802718b7988400a9c8ebf0198f",
    "turkish/banking/fold11.json": "7c19235123b05a4827b73e93e1f94c8e99d95b461fdfbf23d97a9addde224e3e",
    "turkish/banking/fold12.json": "1f08dafd200d76d6f1ba974baf68e92e5108c219e55f1f0749b1891bf3f2d1d9",
    "turkish/banking/fold13.json": "9a978a77167a08d91fdb828e17c67a7da90bbbbcb32defff4de51f787233f143",
    "turkish/banking/fold14.json": "00cafe47bd5ce22a27967bb05dfc31bdbb3d0885e15851398fd6501ab0a41349",
    "turkish/banking/fold15.json": "c61e519624b021017c7f7716de74c90b3e8f76897a34c94437f9a076afbf8914",
    "turkish/banking/fold16.json": "39d6ebdeb73f1d16c2522c228b861c6f000dd9712ef04461cf43d920d1744741",
    "turkish/banking/fold17.json": "78ca8f6421373fb1ada07ac0df3f2abb7313b9cfc94e040e4c54cd74a994e7d2",
    "turkish/banking/fold18.json": "f67f0ce063176eb734babcbc6a51211a437a32ea70abc6c4404328726e89f215",
    "turkish/banking/fold19.json": "f8ec6eae632baa9c39751b0294ddbb8ce7fff922370998e1d146ad16484bb70b",
    "turkish/hotels/fold0.json": "5303e79ecbac7c9546795acc6b1463aa348270bdc5bedfb8f5b11b412ea6907f",
    "turkish/hotels/fold1.json": "1f1ec03760912063429d20e1167329dc69d1591958bf779e5831407fa5ab467e",
    "turkish/hotels/fold2.json": "be68df2ad7b140d718c48403a455b1bd19aa0daabb53a0f1f00c557f9a8b9ed2",
    "turkish/hotels/fold3.json": "952437b958ba43bc1a37d689526842fb52524caea15493d1a710dfa5fc0b5707",
    "turkish/hotels/fold4.json": "8cedd8bb413750142a240cfad1d48dc3752d610cac13e1ffe50182bdcd183504",
    "turkish/hotels/fold5.json": "4ee47383c85e7d0241b9a20c2db375ed3384977c88e1335c053d0da7c8890cd7",
    "turkish/hotels/fold6.json": "8428e8dc3aea472763c8945348688c6b97aeea48e3378f0ee6bbd68b99264823",
    "turkish/hotels/fold7.json": "74f855e41e3d26af0ae29b25977fbd15e811071a3de7a4055dafcf45bdb014e8",
    "turkish/hotels/fold8.json": "79a64be5ba75dc312f817a8b1da44285f27b9b69ae99050c5281db2b62244bb6",
    "turkish/hotels/fold9.json": "0d7afc6c5e07e00af0a0fd73ffca7df8174acfc82150656b664f40984b5fc1f1",
    "turkish/hotels/fold10.json": "041389ce081d916b2fc07e46745bde5e4cedb588b82c43fc28c113e4a30536b9",
    "turkish/hotels/fold11.json": "b107910b342ec582adb43b4f0f3b67372d691b80d3045bed2da5dd1c413dc950",
    "turkish/hotels/fold12.json": "c3af5c4a393319795baae2b60ae0ab92b4624266b819ef1f4c88e669881d578d",
    "turkish/hotels/fold13.json": "88b7452cbef2cba5b8e95c2b372ad3159925656fa4fe8b9b452613b7e4bf50d0",
    "turkish/hotels/fold14.json": "5a8a61c1b25dea1ceede0169a8730f383734dcad9d3a815b1b2528dcedf2635d",
    "turkish/hotels/fold15.json": "64850d26d7a132654ae15046b4befb332ccc5703b783012d89c8fad85e09e5f7",
    "turkish/hotels/fold16.json": "081d8931b3cfb16205ed26061382d909998e948e0bc51fc2b753283f36969e6c",
    "turkish/hotels/fold17.json": "52a36d80272bed64aa6f8366c6917f3268735afbdfa1195b895c44c23de7c661",
    "turkish/hotels/fold18.json": "eb9d5035cfc394ced6280ded921b7edb29e16c169e197c7d4e645f2e62c93e4d",
    "turkish/hotels/fold19.json": "5ddf9bfef59312194661a194a5e465384d3f13824bbc5c921de2a00f2b755c97",
}

# Laya multilingual weights. Do not digest tokenizer/tokenizer_config.json: laya may rewrite it in the snapshot.
LAYA = {
    "repo": "convaiinnovations/laya",
    "subfolder": "multilingual",
    "revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
    "allow_patterns": ["multilingual/rl_agent_config.json", "multilingual/model.safetensors",
                       "multilingual/tokenizer/*", "multilingual/encoder/*"],
    "sha256": {
        "model.safetensors": "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
        "tokenizer/tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
        "rl_agent_config.json": "25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6",
        "encoder/config.json": "83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4",
    },
}

# BERTurk baseline weights: safetensors only, never the pickle/TF/flax files the repo also ships.
BERTURK = {
    "repo": "dbmdz/bert-base-turkish-cased",
    "subfolder": None,
    "revision": "b6e1de16c983e0f2c70664591ea3f22810072608",
    "allow_patterns": ["config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt"],
    "sha256": {
        "config.json": "3be8dbe74813f59a252773db34bd67640166d36c507ded3b06b55f91f05ce78a",
        "model.safetensors": "18e29c6c61a046ab18fa48aeec3dce1285b09c5648d9ba68da1474940b9a9fcd",
        "tokenizer_config.json": "abb7cc8c813870875157e0dff63bc21bb521d4b5d3c35c2eb9facaee4fd9532f",
        "vocab.txt": "ca72d2012b1e46ed42f5df2dc9675f9da0b37a481601785cb4dff91f67fcca65",
    },
}

# The 22 banking-only tags (INTENTS minus tags that also occur in HOTELS): the OOD scoring subset for the probe.
OOD_SCORING_TAGS = [
    "account", "appointment", "arrival", "balance", "business", "card", "cheque", "contactless", "credit",
    "current", "debit", "direct_debit", "fees_interests", "international", "loan", "mortgage", "overdraft",
    "pin", "savings", "standing_order", "transfer_payment_deposit", "withdrawal",
]
