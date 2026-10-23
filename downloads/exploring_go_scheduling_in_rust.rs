//! A bounded green-thread experiment, NOT a runtime for arbitrary Rust closures.
//! Build: rustc -O main.rs -o goroutines (Rust 1.88+; macOS ARM64 only).
//! Run: ./goroutines --help
//!
//! Only the labelled assembly kernel may be asynchronously preempted. It has
//! no Rust frames, allocations, locks, I/O or TLS access. Migrating tasks also
//! contain only assembly frames. Scheduler Rust frames stay on their OS thread.
//! x16 is RESERVED for the resumption PC; x18 is Apple's reserved platform
//! register and is never modified. This is a deliberate kernel ABI, not a
//! promise to suspend arbitrary native instructions safely.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("This experiment requires macOS on ARM64 (Apple Silicon).");

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod demo {
use std::arch::naked_asm;
use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::mem::{offset_of, size_of};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Barrier};
use std::time::Instant;

// At an explicit call boundary, the ABI makes caller-saved registers disposable.
// FPCR/FPSR are included so our test tasks can use different FP environments.
#[repr(C, align(16))]
#[derive(Default)]
struct Context {
    x: [u64; 12],                 // x19..x30
    sp: u64,
    d: [u64; 8],                  // low 64 bits of v8..v15
    fpcr: u64,
    fpsr: u64,
}
const _: () = {
    assert!(size_of::<Context>() == 192);
    assert!(offset_of!(Context, sp) == 96);
    assert!(offset_of!(Context, d) == 104);
    assert!(offset_of!(Context, fpcr) == 168);
    assert!(offset_of!(Context, fpsr) == 176);
};

#[unsafe(naked)]
unsafe extern "C" fn switch_context(_old: *mut Context, _new: *const Context) {
    naked_asm!(
        "stp x19, x20, [x0, #0]", "stp x21, x22, [x0, #16]",
        "stp x23, x24, [x0, #32]", "stp x25, x26, [x0, #48]",
        "stp x27, x28, [x0, #64]", "stp x29, x30, [x0, #80]",
        "mov x2, sp", "str x2, [x0, #96]",
        "stp d8, d9, [x0, #104]", "stp d10, d11, [x0, #120]",
        "stp d12, d13, [x0, #136]", "stp d14, d15, [x0, #152]",
        "mrs x2, fpcr", "mrs x3, fpsr", "stp x2, x3, [x0, #168]",
        "ldp x19, x20, [x1, #0]", "ldp x21, x22, [x1, #16]",
        "ldp x23, x24, [x1, #32]", "ldp x25, x26, [x1, #48]",
        "ldp x27, x28, [x1, #64]", "ldp x29, x30, [x1, #80]",
        "ldr x2, [x1, #96]", "mov sp, x2",
        "ldp d8, d9, [x1, #104]", "ldp d10, d11, [x1, #120]",
        "ldp d12, d13, [x1, #136]", "ldp d14, d15, [x1, #152]",
        "ldp x2, x3, [x1, #168]", "msr fpcr, x2", "msr fpsr, x3",
        "ret",
    );
}

#[repr(C)]
#[derive(Default, Debug, PartialEq, Eq)]
struct Results {
    count: u64,
    sum: u64,
    xor: u64,
    v0: [u64; 2],
    v8: [u64; 2],
    v31: [u64; 2],
    fpcr: u64,
    fpsr: u64,
    red_zone: u64,
}

#[repr(C)]
#[derive(Default)]
struct TaskMemory {
    ctx: Context,                 // MUST be first: also the switch_context pointer
    scheduler: *mut Context,      // refreshed on every dispatch, including migration
    resume_pc: u64,               // signal handler / cooperative yield -> trampoline
    done: u64,
    iterations: u64,
    quantum: u64,                 // 0 means no cooperative yield
    seed: u64,
    results: Results,
    preemptions: u64,
    flag_faults: u64,             // OR of discrepancies observed after restoration
}

// Entry is reached by switch_context, not by a Rust call. Its only caller frame
// is assembly. The BL/RET pair deliberately tests preservation of original LR.
#[unsafe(naked)]
unsafe extern "C" fn task_entry() {
    naked_asm!(
        "mov x0, #0", "ldr x1, [x19, #{seed}]", "mov x2, x1",
        "ldr x4, [x19, #{iterations}]", "ldr x5, [x19, #{quantum}]",
        "mov x6, x5",
        "dup v0.2d, x1", "mov x9, #3", "mul x9, x1, x9", "dup v8.2d, x9",
        "mov x9, #7", "mul x9, x1, x9", "dup v31.2d, x9",
        "mov x9, #1", "dup v1.2d, x9",
        "bl {kernel}",
        "add x9, x19, #{results}",
        "stp x0, x1, [x9, #0]", "str x2, [x9, #16]",
        "str q0, [x9, #24]", "str q8, [x9, #40]", "str q31, [x9, #56]",
        "mrs x10, fpcr", "mrs x11, fpsr", "stp x10, x11, [x9, #72]",
        "mov x9, #1", "str x9, [x19, #{done}]",
        "mov x0, x19", "ldr x1, [x19, #{scheduler}]",
        "bl {switch}", "brk #1",  // a completed task MUST NOT be resumed
        kernel = sym compute_kernel,
        switch = sym switch_context,
        seed = const offset_of!(TaskMemory, seed),
        iterations = const offset_of!(TaskMemory, iterations),
        quantum = const offset_of!(TaskMemory, quantum),
        results = const offset_of!(TaskMemory, results),
        done = const offset_of!(TaskMemory, done),
        scheduler = const offset_of!(TaskMemory, scheduler),
    );
}

// Only instructions in [demo_kernel_begin, demo_kernel_end) can be interrupted.
// In quantum=0 mode there are NO calls, yields, flag polls or syscalls in the loop.
// Flags are live across CMP/B.HS and SUBS/B.NE. SIMD upper halves are also live.
#[unsafe(naked)]
unsafe extern "C" fn compute_kernel() {
    naked_asm!(
        // A live red-zone value checks that injected frames leave it alone.
        "mov x9, #0x5a5a", "stur x9, [sp, #-8]",
        ".globl _demo_kernel_begin", "_demo_kernel_begin:",
        "2:", "cmp x0, x4", "b.hs 4f",
        "add x1, x1, x0", "eor x2, x2, x0", "add x0, x0, #1",
        "add v0.2d, v0.2d, v1.2d", "add v8.2d, v8.2d, v1.2d",
        "cbz x5, 2b", "subs x6, x6, #1", "b.ne 2b",
        // Do not accept asynchronous signals during the cooperative handoff.
        ".globl _demo_kernel_end", "_demo_kernel_end:",
        "mov x6, x5", "adr x16, 2b", "str x16, [x19, #{resume}]",
        "b {suspend}", "4:",
        "ldur x9, [sp, #-8]", "str x9, [x19, #{red_zone}]", "ret",
        red_zone = const offset_of!(TaskMemory, results) + offset_of!(Results, red_zone),
        resume = const offset_of!(TaskMemory, resume_pc),
        suspend = sym suspend_task,
    );
}

// Both cooperative yields and signal-return PC redirection enter here.
// Leave 128 bytes (Darwin red zone) untouched below the interrupted SP.
// 800-byte frame: GPRs 0..159, NZCV/PC 160..175, vectors 192..703.
// x19..x29 and FPCR/FPSR are saved by switch_context; x18 is untouched.
// x16 is intentionally restored as the continuation PC, NOT its previous value.
#[unsafe(naked)]
unsafe extern "C" fn suspend_task() {
    naked_asm!(
        "sub sp, sp, #928",
        "stp x0, x1, [sp, #0]", "stp x2, x3, [sp, #16]",
        "stp x4, x5, [sp, #32]", "stp x6, x7, [sp, #48]",
        "stp x8, x9, [sp, #64]", "stp x10, x11, [sp, #80]",
        "stp x12, x13, [sp, #96]", "stp x14, x15, [sp, #112]",
        "stp x16, x17, [sp, #128]", "str x30, [sp, #144]",
        "mrs x9, nzcv", "ldr x10, [x19, #{resume}]", "stp x9, x10, [sp, #160]",
        "stp q0, q1, [sp, #192]", "stp q2, q3, [sp, #224]",
        "stp q4, q5, [sp, #256]", "stp q6, q7, [sp, #288]",
        "stp q8, q9, [sp, #320]", "stp q10, q11, [sp, #352]",
        "stp q12, q13, [sp, #384]", "stp q14, q15, [sp, #416]",
        "stp q16, q17, [sp, #448]", "stp q18, q19, [sp, #480]",
        "stp q20, q21, [sp, #512]", "stp q22, q23, [sp, #544]",
        "stp q24, q25, [sp, #576]", "stp q26, q27, [sp, #608]",
        "stp q28, q29, [sp, #640]", "stp q30, q31, [sp, #672]",
        "mov x0, x19", "ldr x1, [x19, #{scheduler}]", "bl {switch}",
        "ldp q0, q1, [sp, #192]", "ldp q2, q3, [sp, #224]",
        "ldp q4, q5, [sp, #256]", "ldp q6, q7, [sp, #288]",
        "ldp q8, q9, [sp, #320]", "ldp q10, q11, [sp, #352]",
        "ldp q12, q13, [sp, #384]", "ldp q14, q15, [sp, #416]",
        "ldp q16, q17, [sp, #448]", "ldp q18, q19, [sp, #480]",
        "ldp q20, q21, [sp, #512]", "ldp q22, q23, [sp, #544]",
        "ldp q24, q25, [sp, #576]", "ldp q26, q27, [sp, #608]",
        "ldp q28, q29, [sp, #640]", "ldp q30, q31, [sp, #672]",
        "ldr x9, [sp, #160]", "msr nzcv, x9",
        // Validate flags directly: timer sampling need not hit a flag-sensitive
        // instruction boundary. These instructions do NOT change NZCV.
        "mrs x9, nzcv", "ldr x10, [sp, #160]", "eor x9, x9, x10",
        "ldr x10, [x19, #{flag_faults}]", "orr x9, x9, x10",
        "str x9, [x19, #{flag_faults}]",
        "ldp x0, x1, [sp, #0]",  "ldp x2, x3, [sp, #16]",
        "ldp x4, x5, [sp, #32]", "ldp x6, x7, [sp, #48]",
        "ldp x8, x9, [sp, #64]", "ldp x10, x11, [sp, #80]",
        "ldp x12, x13, [sp, #96]", "ldp x14, x15, [sp, #112]",
        "ldr x17, [sp, #136]", "ldr x30, [sp, #144]",
        "ldr x16, [sp, #168]", "add sp, sp, #928", "br x16",
        resume = const offset_of!(TaskMemory, resume_pc),
        scheduler = const offset_of!(TaskMemory, scheduler),
        flag_faults = const offset_of!(TaskMemory, flag_faults),
        switch = sym switch_context,
    );
}

// Native declarations: checked against the installed SDK by abi-check.c.
unsafe extern "C" {
    static demo_kernel_begin: u8;
    static demo_kernel_end: u8;
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
    fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn getpagesize() -> i32;
    fn sigaction(sig: i32, act: *const SigAction, old: *mut SigAction) -> i32;
    fn sigaltstack(new: *const SignalStack, old: *mut SignalStack) -> i32;
    fn sigprocmask(how: i32, set: *const u32, old: *mut u32) -> i32;
    fn sigpending(set: *mut u32) -> i32;
    fn sigwait(set: *const u32, sig: *mut i32) -> i32;
    fn setitimer(which: i32, new: *const Timer, old: *mut Timer) -> i32;
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Timeval { sec: i64, usec: i32, padding: i32 }
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Timer { interval: Timeval, value: Timeval }
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SigAction { handler: usize, mask: u32, flags: i32 }
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SignalStack { ptr: *mut c_void, size: usize, flags: i32 }
#[repr(C)]
struct UContext {
    onstack: i32, sigmask: u32, stack: SignalStack,
    link: *mut c_void, mcsize: usize, mcontext: *mut u8,
}
const _: () = {
    assert!(offset_of!(UContext, mcontext) == 48);
    assert!(size_of::<Timeval>() == 16);
    assert!(size_of::<SigAction>() == 16);
    assert!(size_of::<SignalStack>() == 24);
};
const SIGALRM: i32 = 14;
const ALARM_MASK: u32 = 1 << (SIGALRM - 1);
static SIGNALS: AtomicU64 = AtomicU64::new(0);
static ACCEPTED: AtomicU64 = AtomicU64::new(0);

extern "C" fn timer_handler(_: i32, _: *mut c_void, uc: *mut UContext) {
    // No allocation, I/O, locks, references into Runtime, or scheduler calls.
    // Signal return leaves the handler/alternate stack before switching tasks.
    unsafe {
        SIGNALS.fetch_add(1, Ordering::Relaxed);
        let mc = (*uc).mcontext;
        let pc = mc.add(272).cast::<u64>();
        let value = *pc as usize;
        let begin = ptr::addr_of!(demo_kernel_begin) as usize;
        let end = ptr::addr_of!(demo_kernel_end) as usize;
        if value < begin || value >= end { return; }
        // x19 is a valid TaskMemory pointer ONLY within this labelled kernel.
        let task = *(mc.add(16 + 19 * 8).cast::<*mut TaskMemory>());
        (*task).resume_pc = *pc;
        (*task).preemptions += 1;
        *pc = suspend_task as *const () as u64;
        ACCEPTED.fetch_add(1, Ordering::Relaxed);
    }
}

struct GuardedStack { base: *mut c_void, len: usize, usable: *mut c_void, bytes: usize }
impl GuardedStack {
    fn new(bytes: usize) -> Self {
        unsafe {
            let page = getpagesize() as usize;
            assert!(page.is_power_of_two());
            let bytes = (bytes + page - 1) & !(page - 1);
            let len = bytes + 2 * page;
            let base = mmap(ptr::null_mut(), len, 0, 0x1002, -1, 0);
            assert_ne!(base as isize, -1, "mmap: {}", std::io::Error::last_os_error());
            let usable = base.cast::<u8>().add(page).cast::<c_void>();
            if mprotect(usable, bytes, 3) != 0 {
                let error = std::io::Error::last_os_error();
                munmap(base, len);
                panic!("mprotect: {}", error);
            }
            Self { base, len, usable, bytes }
        }
    }
    fn top(&self) -> u64 { self.usable as u64 + self.bytes as u64 }
}
impl Drop for GuardedStack {
    fn drop(&mut self) { unsafe { assert_eq!(munmap(self.base, self.len), 0, "munmap"); } }
}

struct TimerGuard {
    old_action: SigAction, old_stack: SignalStack, old_mask: u32,
    _stack: GuardedStack,
}
fn check(rc: i32, operation: &str) {
    assert_eq!(rc, 0, "{operation}: {}", std::io::Error::last_os_error());
}
impl TimerGuard {
    fn start(micros: u64) -> Self {
        unsafe {
            let mut old_mask = 0;
            check(sigprocmask(1, &ALARM_MASK, &mut old_mask), "block SIGALRM");
            assert_eq!(old_mask & ALARM_MASK, 0, "SIGALRM was already blocked");
            let mut old_timer = Timer::default();
            check(setitimer(0, &Timer::default(), &mut old_timer), "inspect timer");
            assert_eq!(old_timer.value.sec | old_timer.value.usec as i64, 0, "existing ITIMER_REAL timer");
            let stack = GuardedStack::new(128 * 1024);
            let new_stack = SignalStack { ptr: stack.usable, size: stack.bytes, flags: 0 };
            let mut old_stack = SignalStack::default();
            check(sigaltstack(&new_stack, &mut old_stack), "sigaltstack");
            let action = SigAction { handler: timer_handler as *const () as usize, mask: ALARM_MASK, flags: 0x0041 };
            let mut old_action = SigAction::default();
            check(sigaction(SIGALRM, &action, &mut old_action), "sigaction");
            SIGNALS.store(0, Ordering::Relaxed);
            ACCEPTED.store(0, Ordering::Relaxed);
            let t = Timeval { sec: (micros / 1_000_000) as i64, usec: (micros % 1_000_000) as i32, padding: 0 };
            check(setitimer(0, &Timer { interval: t, value: t }, ptr::null_mut()), "start timer");
            check(sigprocmask(3, &old_mask, ptr::null_mut()), "unblock SIGALRM");
            Self { old_action, old_stack, old_mask, _stack: stack }
        }
    }
}
impl Drop for TimerGuard {
    fn drop(&mut self) {
        unsafe {
            check(sigprocmask(1, &ALARM_MASK, ptr::null_mut()), "block SIGALRM");
            check(setitimer(0, &Timer::default(), ptr::null_mut()), "stop timer");
            // Drain a pending timer signal before restoring its old disposition.
            let mut pending = 0;
            check(sigpending(&mut pending), "sigpending");
            if pending & ALARM_MASK != 0 {
                let mut sig = 0;
                check(sigwait(&ALARM_MASK, &mut sig), "sigwait");
                assert_eq!(sig, SIGALRM);
            }
            check(sigaction(SIGALRM, &self.old_action, ptr::null_mut()), "restore action");
            check(sigaltstack(&self.old_stack, ptr::null_mut()), "restore signal stack");
            check(sigprocmask(3, &self.old_mask, ptr::null_mut()), "restore signal mask");
        }
    }
}

struct Task {
    // Stable address: this box and stack may move between owners, not their data.
    memory: Box<UnsafeCell<TaskMemory>>,
    _stack: GuardedStack,
    id: usize,
    last_worker: Option<usize>,
    migrations: u64,
    slices: u64,
}
// SAFETY: only one worker owns a Task. Channels transfer exclusive ownership.
// Its suspended frames are assembly only, with no TLS or thread-affine values.
// The scheduler pointer is replaced before resuming on a different OS worker.
unsafe impl Send for Task {}
impl Task {
    fn new(id: usize, iterations: u64, quantum: u64) -> Self {
        let stack = GuardedStack::new(64 * 1024);
        let memory = Box::new(UnsafeCell::new(TaskMemory::default()));
        let p = memory.get();
        unsafe {
            (*p).ctx.sp = stack.top();
            (*p).ctx.x[0] = p as u64; // x19
            (*p).ctx.x[11] = task_entry as *const () as u64; // LR
            (*p).ctx.fpcr = ((id % 4) as u64) << 22;
            (*p).ctx.fpsr = (id % 2) as u64 * 16;
            (*p).iterations = iterations;
            (*p).quantum = quantum;
            (*p).seed = id as u64 + 17;
        }
        Self { memory, _stack: stack, id, last_worker: None, migrations: 0, slices: 0 }
    }
    fn dispatch(&mut self, scheduler: *mut Context, worker: usize) -> bool {
        if self.last_worker.is_some_and(|last| last != worker) { self.migrations += 1; }
        self.last_worker = Some(worker);
        self.slices += 1;
        // No &TaskMemory or &mut Context survives this call. UnsafeCell covers
        // modifications by assembly and signal handlers, including at -O/LTO.
        unsafe {
            let p = self.memory.get();
            assert_eq!((*p).done, 0, "resumed a completed task");
            (*p).scheduler = scheduler;
            switch_context(scheduler, ptr::addr_of!((*p).ctx));
            (*p).scheduler = ptr::null_mut();
            (*p).done == 1
        }
    }
    fn preemptions(&self) -> u64 { unsafe { (*self.memory.get()).preemptions } }
    fn progress(&self) -> u64 {
        unsafe {
            let m = &*self.memory.get();
            // Suspended SP points to the trampoline frame; its first word is x0.
            if m.done == 1 { m.results.count } else { *(m.ctx.sp as *const u64) }
        }
    }
    fn verify(&self) {
        // Task is suspended/finished and exclusively owned here.
        let m = unsafe { &*self.memory.get() };
        let n = m.iterations;
        let seed = m.seed;
        let sum = ((n as u128 * (n - 1) as u128) / 2) as u64;
        let k = n - 1;
        let xor = match k % 4 { 0 => k, 1 => 1, 2 => k + 1, _ => 0 };
        let expected = Results {
            count: n, sum: seed.wrapping_add(sum), xor: seed ^ xor,
            v0: [seed.wrapping_add(n); 2],
            v8: [seed.wrapping_mul(3).wrapping_add(n); 2],
            v31: [seed.wrapping_mul(7); 2],
            fpcr: ((self.id % 4) as u64) << 22,
            fpsr: (self.id % 2) as u64 * 16,
            red_zone: 0x5a5a,
        };
        assert_eq!(m.done, 1, "task {} incomplete", self.id);
        assert_eq!(m.flag_faults, 0, "task {} corrupted: NZCV restoration", self.id);
        assert_eq!(m.results, expected, "task {} corrupted", self.id);
    }
}

#[derive(Clone)]
struct Config {
    mode: String, iterations: u64, tasks: usize, workers: usize,
    timer_us: u64, quantum: u64, rounds: usize, trace: bool,
}
fn single(c: &Config, mode: &str) {
    let quantum = if mode == "cooperative" { c.quantum } else { 0 };
    let mut queue: VecDeque<_> = (0..c.tasks).map(|id| Task::new(id, c.iterations, quantum)).collect();
    let mut scheduler = Context::default();
    let mut finished = 0;
    let mut slices = 0;
    let mut interrupted_tasks = 0;
    let mut preemptions = 0;
    let mut incomplete_before_first_completion = 0;
    let start = Instant::now();
    let timer = if mode == "preempt" { Some(TimerGuard::start(c.timer_us)) } else { None };
    while let Some(mut task) = queue.pop_front() {
        let done = task.dispatch(&raw mut scheduler, 0);
        slices += 1;
        if c.trace && slices <= 32 {
            println!("  slice={slices} task={} progress={} done={done} preemptions={}", task.id, task.progress(), task.preemptions());
        }
        if done {
            task.verify();
            let p = task.preemptions();
            interrupted_tasks += usize::from(p > 0);
            preemptions += p;
            finished += 1;
        } else {
            if finished == 0 { incomplete_before_first_completion += 1; }
            queue.push_back(task);
        }
    }
    drop(timer); // stopped/drained before printing or spawning workers
    assert_eq!(finished, c.tasks);
    if mode == "preempt" {
        assert_eq!(interrupted_tasks, c.tasks, "not every task was preempted: increase --iterations or reduce --timer-us");
        assert_eq!(preemptions, ACCEPTED.load(Ordering::Relaxed));
        assert_eq!(slices, c.tasks as u64 + preemptions);
        assert!(incomplete_before_first_completion >= c.tasks, "no complete interleaving round before first completion");
    } else {
        assert_eq!(preemptions, 0);
        if mode == "serial" { assert_eq!(slices, c.tasks as u64); }
        if mode == "cooperative" {
            assert_eq!(slices, c.tasks as u64 * (1 + c.iterations / c.quantum));
        }
    }
    println!("PASS mode={mode} tasks={finished} iterations={} slices={slices} preemptions={preemptions} interrupted_tasks={interrupted_tasks} before_first_completion={incomplete_before_first_completion} elapsed_ms={:.3}", c.iterations, start.elapsed().as_secs_f64() * 1000.0);
}

// A dispatcher transfers exclusive ownership through channels. Round-robin
// handoff deliberately forces actual migration (rather than hoping a shared
// queue happens to move tasks). No lock or Rust frame is suspended in a task.
fn migration(c: &Config) {
    let (result_tx, result_rx) = mpsc::channel();
    let barrier = Arc::new(Barrier::new(c.workers + 1));
    let mut senders = Vec::new();
    let mut handles = Vec::new();
    for worker in 0..c.workers {
        let (tx, rx) = mpsc::channel::<Task>();
        senders.push(tx);
        let result_tx = result_tx.clone();
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            let mut scheduler = Context::default();
            barrier.wait();
            while let Ok(mut task) = rx.recv() {
                let done = task.dispatch(&raw mut scheduler, worker);
                if result_tx.send((worker, task, done)).is_err() { break; }
            }
        }));
    }
    drop(result_tx);
    let start = Instant::now();
    barrier.wait();
    for id in 0..c.tasks {
        senders[id % c.workers].send(Task::new(id, c.iterations, c.quantum)).unwrap();
    }
    let mut completed = 0;
    let mut migrations = 0;
    let mut slices = 0;
    while completed < c.tasks {
        let (worker, task, done) = result_rx.recv().expect("all workers disappeared");
        if done {
            task.verify();
            assert_eq!(task.preemptions(), 0);
            let expected_slices = 1 + c.iterations / c.quantum;
            assert_eq!(task.slices, expected_slices);
            assert_eq!(task.migrations, if c.workers > 1 { expected_slices - 1 } else { 0 });
            migrations += task.migrations;
            slices += task.slices;
            completed += 1;
        } else {
            senders[(worker + 1) % c.workers].send(task).unwrap();
        }
    }
    drop(senders);
    for h in handles { h.join().expect("worker panicked"); }
    assert!(c.workers == 1 || migrations > 0, "workload did not demonstrate migration");
    println!("PASS mode=migrate tasks={completed} workers={} iterations={} slices={slices} migrations={migrations} elapsed_ms={:.3}", c.workers, c.iterations, start.elapsed().as_secs_f64() * 1000.0);
}

fn config() -> Config {
    let mut c = Config { mode: "all".into(), iterations: 5_000_000, tasks: 3, workers: 3, timer_us: 1000, quantum: 50_000, rounds: 1, trace: false };
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--trace" { c.trace = true; continue; }
        if key == "--help" {
            println!("Bounded macOS ARM64 green-thread experiment (not an arbitrary Rust runtime)\n\
Usage: goroutines [--mode all|serial|cooperative|preempt|migrate] [options]\n\
  --iterations N  loop iterations per task (default 5000000)\n\
  --tasks N       tasks (default 3)\n\
  --workers N     migration workers (default 3)\n\
  --timer-us N    preemption timer interval, microseconds (default 1000)\n\
  --quantum N     cooperative iterations between yields (default 50000)\n\
  --rounds N      repeat in this process (default 1)\n\
  --trace         print the first 32 single-worker scheduling events\n\
\nAll results are checked. Preempt mode FAILS if every task was not interrupted.\n\
For 10000-task migration stress: --mode migrate --tasks 10000 --workers 8 --iterations 10000 --quantum 1000\n\
Use an external timeout for fault experiments. No unsafe negative test runs by default.");
            std::process::exit(0);
        }
        let value = args.next().unwrap_or_else(|| panic!("missing value for {}", key));
        match key.as_str() {
            "--mode" => c.mode = value,
            "--iterations" => c.iterations = value.parse().expect("iterations must be an integer"),
            "--tasks" => c.tasks = value.parse().expect("tasks must be an integer"),
            "--workers" => c.workers = value.parse().expect("workers must be an integer"),
            "--timer-us" => c.timer_us = value.parse().expect("timer-us must be an integer"),
            "--quantum" => c.quantum = value.parse().expect("quantum must be an integer"),
            "--rounds" => c.rounds = value.parse().expect("rounds must be an integer"),
            _ => panic!("unknown option {}; use --help", key),
        }
    }
    assert!(["all", "serial", "cooperative", "preempt", "migrate"].contains(&c.mode.as_str()), "unknown mode");
    assert!((1..=10_000_000_000).contains(&c.iterations));
    assert!((1..=100_000).contains(&c.tasks));
    assert!((1..=64).contains(&c.workers));
    assert!((1..=1_000_000).contains(&c.timer_us));
    assert!(c.quantum > 0);
    assert!((1..=100_000).contains(&c.rounds));
    c
}

pub fn run() {
    let c = config();
    println!("Bounded kernel demo: macOS ARM64; x16 reserved; no arbitrary Rust preemption");
    for round in 1..=c.rounds {
        println!("ROUND {round}/{}", c.rounds);
        if c.mode == "all" {
            single(&c, "serial");
            single(&c, "cooperative");
            single(&c, "preempt");
            migration(&c);
        } else if c.mode == "migrate" {
            migration(&c);
        } else {
            single(&c, &c.mode);
        }
    }
    println!("PASS all requested checks (empirical evidence, not a proof of soundness)");
}
}

fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    demo::run();
}
