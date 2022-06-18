import _OSShim

#if os(Linux)

  typealias os_unfair_lock_s = pthread_mutex_t

  func os_unfair_lock() -> os_unfair_lock_s {
    var lock = os_unfair_lock_s()
    pthread_mutex_init(&lock, nil)
    return lock
  }

  func os_unfair_lock_lock(_ lock: UnsafeMutablePointer<os_unfair_lock_s>) {
    pthread_mutex_lock(lock)
  }

  func os_unfair_lock_unlock(_ lock: UnsafeMutablePointer<os_unfair_lock_s>) {
    pthread_mutex_unlock(lock)
  }

#endif
