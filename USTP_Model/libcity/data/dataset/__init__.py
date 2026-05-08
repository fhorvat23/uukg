from libcity.data.dataset.abstract_dataset import AbstractDataset

__all__ = ["AbstractDataset"]


def _register_optional(module_name, class_name):
    """Import optional dataset classes only when local files exist."""
    try:
        module = __import__(module_name, fromlist=[class_name])
        globals()[class_name] = getattr(module, class_name)
        __all__.append(class_name)
    except (ModuleNotFoundError, AttributeError):
        # Some forks keep only a subset of dataset modules.
        pass


_register_optional("libcity.data.dataset.trajectory_dataset", "TrajectoryDataset")
_register_optional("libcity.data.dataset.traffic_state_datatset", "TrafficStateDataset")
_register_optional("libcity.data.dataset.traffic_state_cpt_dataset", "TrafficStateCPTDataset")
_register_optional("libcity.data.dataset.traffic_state_point_dataset", "TrafficStatePointDataset")
_register_optional("libcity.data.dataset.traffic_state_grid_dataset", "TrafficStateGridDataset")
_register_optional("libcity.data.dataset.traffic_state_grid_od_dataset", "TrafficStateGridOdDataset")
_register_optional("libcity.data.dataset.traffic_state_od_dataset", "TrafficStateOdDataset")
_register_optional("libcity.data.dataset.eta_dataset", "ETADataset")
_register_optional("libcity.data.dataset.map_matching_dataset", "MapMatchingDataset")
_register_optional("libcity.data.dataset.roadnetwork_dataset", "RoadNetWorkDataset")
_register_optional("libcity.data.dataset.hgcn_dataset", "HGCNDataset")
