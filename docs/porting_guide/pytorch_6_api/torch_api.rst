.. _PyTorch接口替换:

PyTorch接口替换
--------------------------

.. list-table:: torch.mlu 接口支持列表
    :widths: 10 75 75 25
    :header-rows: 1

    * - 序号
      - PyTorch API
      - Cambricon PyTorch API
      - 支持情况
    
    * - 1
      - ``torch.cuda.current_blas_handle``
      - ``torch.mlu.current_blas_handle``
      - 否

    * - 2 
      - ``torch.cuda.current_device``
      - ``torch.mlu.current_device``
      - 是

    * - 3
      - ``torch.cuda.current_stream``
      - ``torch.mlu.current_stream``
      - 是

    * - 4
      - ``torch.cuda.default_stream``
      - ``torch.mlu.default_stream``
      - 是

    * - 5
      - ``torch.cuda.device``
      - ``torch.mlu.device``
      - 是

    * - 6
      - ``torch.cuda.device_count``
      - ``torch.mlu.device_count``
      - 是

    * - 7
      - ``torch.cuda.device_of``
      - ``torch.mlu.device_of``
      - 是

    * - 8 
      - ``torch.cuda.get_device_capability``
      - ``torch.mlu.get_device_capability``
      - 是

    * - 9
      - ``torch.cuda.get_device_name``
      - ``torch.mlu.get_device_name``
      - 是

    * - 10
      - ``torch.cuda.init``
      - ``torch.mlu.init``
      - 是

    * - 11
      - ``torch.cuda.ipc_collect``
      - ``torch.mlu.ipc_collect``
      - 否

    * - 12
      - ``torch.cuda.is_available``
      - ``torch.mlu.is_available``
      - 是

    * - 13
      - ``torch.cuda.is_initialized``
      - ``torch.mlu.is_initialized``
      - 是

    * - 14 
      - ``torch.cuda.set_device``
      - ``torch.mlu.set_device``
      - 是

    * - 15
      - ``torch.cuda.stream``
      - ``torch.mlu.stream``
      - 否

    * - 16
      - ``torch.cuda.synchronize``
      - ``torch.mlu.synchronize``
      - 是

    * - 17
      - ``torch.cuda.get_rng_state``
      - ``torch.mlu.get_rng_state``
      - 否

    * - 18
      - ``torch.cuda.get_rng_state_all``
      - ``torch.mlu.get_rng_state_all``
      - 否

    * - 19
      - ``torch.cuda.set_rng_state``
      - ``torch.mlu.set_rng_state``
      - 否

    * - 20
      - ``torch.cuda.set_rng_state_all``
      - ``torch.mlu.set_rng_state_all``
      - 否

    * - 21
      - ``torch.cuda.manual_seed``
      - ``torch.mlu.manual_seed``
      - 是

    * - 22
      - ``torch.cuda.manual_seed_all``
      - ``torch.mlu.manual_seed_all``
      - 是

    * - 23
      - ``torch.cuda.seed``
      - ``torch.mlu.seed``
      - 否

    * - 24
      - ``torch.cuda.seed_all``
      - ``torch.mlu.seed_all``
      - 否

    * - 25 
      - ``torch.cuda.initial_seed``
      - ``torch.mlu.initial_seed``
      - 否

    * - 26
      - ``torch.cuda.comm.broadcast``
      - ``torch.mlu.comm.broadcast``
      - 否

    * - 27
      - ``torch.cuda.comm.``
        
        ``broadcast_coalesced``
      - ``torch.mlu.comm.``

        ``broadcast_coalesced``
      - 否

    * - 28
      - ``torch.cuda.comm.reduce_add``
      - ``torch.mlu.comm.reduce_add``
      - 否

    * - 29
      - ``torch.cuda.comm.scatter``
      - ``torch.mlu.comm.scatter``
      - 否

    * - 30
      - ``torch.cuda.comm.gather``
      - ``torch.mlu.comm.gather``
      - 否

    * - 31 
      - ``torch.cuda.Stream``
      - ``torch.mlu.Stream``
      - 否

    * - 32
      - ``torch.cuda.Stream.query``
      - ``torch.mlu.Stream.query``
      - 是

    * - 33 
      - ``torch.cuda.Stream.record_event``
      - ``torch.mlu.Stream.record_event``
      - 否

    * - 34
      - ``torch.cuda.Stream.synchronize``
      - ``torch.mlu.Stream.synchronize``
      - 是

    * - 35
      - ``torch.cuda.Stream.wait_event``
      - ``torch.mlu.Stream.wait_event``
      - 否

    * - 36
      - ``torch.cuda.Stream.wait_stream``
      - ``torch.mlu.Stream.wait_stream``
      - 否

    * - 37
      - ``torch.cuda.Event``
      - ``torch.mlu.Event``
      - 否

    * - 38
      - ``torch.cuda.Event.elapsed_time``
      - ``torch.mlu.Event.elapsed_time``
      - 否

    * - 39
      - ``torch.cuda.Event.from_ipc_handle``
      - ``torch.mlu.Event.from_ipc_handle``
      - 否

    * - 40
      - ``torch.cuda.Event.ipc_handle``
      - ``torch.mlu.Event.ipc_handle``
      - 否

    * - 41 
      - ``torch.cuda.Event.query``
      - ``torch.mlu.Event.query``
      - 否

    * - 42
      - ``torch.cuda.Event.record``
      - ``torch.mlu.Event.record``
      - 否

    * - 43
      - ``torch.cuda.Event.synchronize``
      - ``torch.mlu.Event.synchronize``
      - 否

    * - 44 
      - ``torch.cuda.Event.wait``
      - ``torch.mlu.Event.wait``
      - 否

    * - 45 
      - ``torch.cuda.empty_cache``
      - ``torch.mlu.empty_cache``
      - 是

    * - 46
      - ``torch.cuda.memory_stats``
      - ``torch.mlu.memory_stats``
      - 否

    * - 47
      - ``torch.cuda.memory_summary``
      - ``torch.mlu.memory_summary``
      - 否

    * - 48 
      - ``torch.cuda.memory_snapshot``
      - ``torch.mlu.memory_snapshot``
      - 否

    * - 49
      - ``torch.cuda.memory_allocated``
      - ``torch.mlu.memory_allocated``
      - 是

    * - 50
      - ``torch.cuda.max_memory_allocated``
      - ``torch.mlu.max_memory_allocated``
      - 是

    * - 51
      - ``torch.cuda.``

        ``reset_max_memory_allocated``
      - ``torch.mlu.``

        ``reset_max_memory_allocated``
      - 是

    * - 52
      - ``torch.cuda.memory_reserved``
      - ``torch.mlu.memory_reserved``
      - 是

    * - 53
      - ``torch.cuda.max_memory_reserved``
      - ``torch.mlu.max_memory_reserved``
      - 是

    * - 54
      - ``torch.cuda.memory_cached``
      - ``torch.mlu.memory_cached``
      - 是

    * - 55
      - ``torch.cuda.max_memory_cached``
      - ``torch.mlu.max_memory_cached``
      - 是

    * - 56
      - ``torch.cuda.``

        ``reset_max_memory_cached``
      - ``torch.mlu.``

        ``reset_max_memory_cached``
      - 是

    * - 57
      - ``torch.cuda.nvtx.mark``
      - ``torch.mlu.nvtx.mark``
      - 否

    * - 58 
      - ``torch.cuda.nvtx.range_push``
      - ``torch.mlu.nvtx.range_push``
      - 否

    * - 59
      - ``torch.cuda.nvtx.range_pop``
      - ``torch.mlu.nvtx.range_pop``
      - 否

    * - 60
      - ``torch.cuda._sleep``
      - ``torch.mlu._sleep``
      - 否

    * - 61
      - ``torch.cuda.Stream.priority_range``
      - ``torch.mlu.Stream.priority_range``
      - 否

    * - 62
      - ``torch.cuda.get_device_properties``
      - ``torch.mlu.get_device_properties``
      - 是

    * - 63
      - ``torch.cuda.amp.GradScaler``
      - ``torch.mlu.amp.GradScaler``
      - 是

