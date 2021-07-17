from pytracking.evaluation import Tracker, get_dataset, trackerlist


def got10k():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('brost', 'brost', range(1))

    dataset = get_dataset('got10k_test')
    return trackers, dataset


def otb():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('brost', 'brost', range(1))

    dataset = get_dataset('otb')
    return trackers, dataset

def trackingnet():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('brost', 'brost', range(1))

    dataset = get_dataset('trackingnet')
    return trackers, dataset

def uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('brost', 'brost', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset

def lasot():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('brost', 'brost', range(1))

    dataset = get_dataset('lasot')
    return trackers, dataset

def tpl():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('brost', 'brost', range(1))

    dataset = get_dataset('tpl')
    return trackers, dataset

def nfs():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('brost', 'brost', range(1))

    dataset = get_dataset('nfs')
    return trackers, dataset
