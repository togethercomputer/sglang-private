from __future__ import annotations

import msgspec


class SpecTrainingInfo(msgspec.Struct):
    """
    Tracks spec training info for requests in a batch.

    Keys:
    - data_ids: rid -> data_id mapping
    - packed_loss_masks: data_id -> packed_loss_mask string
    - mooncake_store_keys: data_id -> list of keys
    """

    data_ids: dict[str, str] = msgspec.field(default_factory=dict)
    packed_loss_masks: dict[str, str | None] = msgspec.field(default_factory=dict)
    mooncake_store_keys: dict[str, list[str]] = msgspec.field(default_factory=dict)

    def add_request(
        self,
        rid: str,
        data_id: str | None,
        packed_loss_mask: str | None,
    ) -> None:
        """
        Add metadata for a speculative-training request.

        """
        if data_id is not None:
            self.data_ids[rid] = data_id
            self.packed_loss_masks[data_id] = packed_loss_mask
            if data_id not in self.mooncake_store_keys:
                self.mooncake_store_keys[data_id] = []

    def has_request(self, rid: str) -> bool:
        return rid in self.data_ids

    def remove_request(self, rid: str) -> None:
        data_id = self.data_ids.pop(rid, None)
        if data_id is not None:
            remaining_rids_with_data_id = [
                r for r, d in self.data_ids.items() if d == data_id
            ]
            if not remaining_rids_with_data_id:
                self.packed_loss_masks.pop(data_id, None)
                self.mooncake_store_keys.pop(data_id, None)

    def is_empty(self) -> bool:
        return not self.data_ids
