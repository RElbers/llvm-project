; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx950 < %s | FileCheck %s

; Waits between LDS DMA writes and later LDS accesses when the DMA's LDS
; destination does not carry an alias scope. The module LDS lowering gives the
; LDS accesses of a kernel that uses more than one LDS variable alias scopes,
; but only where the pointer is derived from the variable. A destination built
; through an integer has none, so the DMA may write any LDS variable.

@lds.0 = internal addrspace(3) global [256 x i32] poison, align 16
@lds.1 = internal addrspace(3) global [256 x i32] poison, align 16

declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3) nocapture, i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.wave.barrier()

; The read has a scope, the DMA writing what it reads does not.
define amdgpu_kernel void @unscoped_dma_scoped_read(ptr addrspace(8) %rsrc, i32 %off, ptr addrspace(1) %out) {
; CHECK-LABEL: {{^}}unscoped_dma_scoped_read:
; CHECK:       buffer_load_dword {{.*}} lds
; CHECK-NOT:   ds_read
; CHECK:       s_waitcnt vmcnt(0)
; CHECK-NEXT:  ds_read_b32
  %base = ptrtoint ptr addrspace(3) @lds.0 to i32
  %addr = add i32 %base, %off
  %dst = inttoptr i32 %addr to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %dst, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.wave.barrier()
  %val = load i32, ptr addrspace(3) @lds.0
  store i32 0, ptr addrspace(3) @lds.1
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; The read must wait for the unscoped DMA too, not only for the older DMA
; with its scope.
define amdgpu_kernel void @scoped_dma_then_unscoped_dma(ptr addrspace(8) %rsrc, i32 %off, ptr addrspace(1) %out) {
; CHECK-LABEL: {{^}}scoped_dma_then_unscoped_dma:
; CHECK:       buffer_load_dword {{.*}} lds
; CHECK:       buffer_load_dword {{.*}} lds
; CHECK-NOT:   ds_read
; CHECK:       s_waitcnt vmcnt(0)
; CHECK-NEXT:  ds_read_b32
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) @lds.1, i32 4, i32 0, i32 0, i32 0, i32 0)
  %base = ptrtoint ptr addrspace(3) @lds.0 to i32
  %addr = add i32 %base, %off
  %dst = inttoptr i32 %addr to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %dst, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.wave.barrier()
  %val = load i32, ptr addrspace(3) @lds.1
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; The read waits for the unscoped DMA but not for the younger DMA into the
; other variable.
define amdgpu_kernel void @unscoped_dma_then_scoped_dma(ptr addrspace(8) %rsrc, i32 %off, ptr addrspace(1) %out) {
; CHECK-LABEL: {{^}}unscoped_dma_then_scoped_dma:
; CHECK:       buffer_load_dword {{.*}} lds
; CHECK:       buffer_load_dword {{.*}} lds
; CHECK-NOT:   ds_read
; CHECK:       s_waitcnt vmcnt(1)
; CHECK-NEXT:  ds_read_b32
  %base = ptrtoint ptr addrspace(3) @lds.0 to i32
  %addr = add i32 %base, %off
  %dst = inttoptr i32 %addr to ptr addrspace(3)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %dst, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) @lds.1, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.wave.barrier()
  %val = load i32, ptr addrspace(3) @lds.0
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; Two DMAs into the same variable at disjoint offsets. All DMAs with the same
; scope are tracked in one slot, which holds the score of the latest of them,
; so a read overlapping only the second one must still wait for it.
define amdgpu_kernel void @read_from_later_dma_same_scope(ptr addrspace(8) %rsrc, ptr addrspace(1) %out) {
; CHECK-LABEL: {{^}}read_from_later_dma_same_scope:
; CHECK:       buffer_load_dword {{.*}} lds
; CHECK:       buffer_load_dword {{.*}} lds
; CHECK-NOT:   ds_read
; CHECK:       s_waitcnt vmcnt(0)
; CHECK-NEXT:  ds_read_b32
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) @lds.0, i32 4, i32 0, i32 0, i32 0, i32 0)
  %dst = getelementptr i8, ptr addrspace(3) @lds.0, i32 256
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %dst, i32 4, i32 256, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.wave.barrier()
  %val = load i32, ptr addrspace(3) %dst
  store i32 0, ptr addrspace(3) @lds.1
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; Same, with a younger DMA into the other variable that the read need not
; wait for.
define amdgpu_kernel void @read_from_later_dma_same_scope_other_pending(ptr addrspace(8) %rsrc, ptr addrspace(1) %out) {
; CHECK-LABEL: {{^}}read_from_later_dma_same_scope_other_pending:
; CHECK:       buffer_load_dword {{.*}} lds
; CHECK:       buffer_load_dword {{.*}} lds
; CHECK:       buffer_load_dword {{.*}} lds
; CHECK-NOT:   ds_read
; CHECK:       s_waitcnt vmcnt(1)
; CHECK-NEXT:  ds_read_b32
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) @lds.0, i32 4, i32 0, i32 0, i32 0, i32 0)
  %dst = getelementptr i8, ptr addrspace(3) @lds.0, i32 256
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %dst, i32 4, i32 256, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) @lds.1, i32 4, i32 512, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.wave.barrier()
  %val = load i32, ptr addrspace(3) %dst
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; The read is disjoint from the only DMA into the variable, but the slot does
; not record which part of the variable its DMAs wrote, so it waits anyway.
define amdgpu_kernel void @disjoint_read_same_scope(ptr addrspace(8) %rsrc, ptr addrspace(1) %out) {
; CHECK-LABEL: {{^}}disjoint_read_same_scope:
; CHECK:       buffer_load_dword {{.*}} lds
; CHECK-NOT:   ds_read
; CHECK:       s_waitcnt vmcnt(0)
; CHECK-NEXT:  ds_read_b32
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) @lds.0, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.wave.barrier()
  %src = getelementptr i8, ptr addrspace(3) @lds.0, i32 512
  %val = load i32, ptr addrspace(3) %src
  store i32 0, ptr addrspace(3) @lds.1
  store i32 %val, ptr addrspace(1) %out
  ret void
}
