


"""
# Paths to your decoded + metadata files
from set_hour_and_batch_TTE_ import run_set_hour_and_batch_TTE
decoded = "/Users/mirandahulsey-vincent/Desktop/analysis_results/R07_RC3_Comp2/TweetyBERT_Pretrain_LLB_AreaX_FallSong_R07_RC3_Comp2_decoded_database.json"
meta    = "/Users/mirandahulsey-vincent/Desktop/analysis_results/R07_RC3_Comp2/R07_RC3_Comp2_metadata.json"

_ = run_set_hour_and_batch_TTE(
    decoded_database_json=decoded,
    creation_metadata_json=meta,
    range1="05:00-07:00",
    range2="14:00-17:00",
    batch_size=25,
    only_song_present=True,
    save_dir=Path("./figures"),
    show=True,
)

"""