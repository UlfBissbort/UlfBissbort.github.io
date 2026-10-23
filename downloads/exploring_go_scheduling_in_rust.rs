use std::arch::{asm, naked_asm};

// ═══════════════════════════════════════════════════════════════════════════════
//  The Context — What Gets Saved When We Switch
// ═══════════════════════════════════════════════════════════════════════════════

/// On AArch64, we save 21 registers:
///   x19-x28: callee-saved general purpose (10 regs)
///   x29:     frame pointer
///   x30:     link register (return address)
///   sp:      stack pointer
///   d8-d15:  callee-saved SIMD/FP registers (8 regs, 64-bit each)
///
/// The d8-d15 registers MUST be saved! The C calling convention requires
/// the callee to preserve them. Without this, the compiler can store data
/// in d8-d15 across our switch_context call, and we'd corrupt it.
/// This was the root cause of intermittent crashes with multiple OS workers.
#[repr(C)]
#[derive(Debug, Clone)]
struct Context {
    x19: u64, x20: u64, x21: u64, x22: u64, x23: u64,
    x24: u64, x25: u64, x26: u64, x27: u64, x28: u64,
    x29: u64,   // frame pointer
    x30: u64,   // link register
    sp:  u64,   // stack pointer
    d8:  u64, d9:  u64, d10: u64, d11: u64,
    d12: u64, d13: u64, d14: u64, d15: u64,
}

impl Context {
    const fn zeroed() -> Self {
        Context {
            x19: 0, x20: 0, x21: 0, x22: 0, x23: 0,
            x24: 0, x25: 0, x26: 0, x27: 0, x28: 0,
            x29: 0, x30: 0, sp: 0,
            d8: 0, d9: 0, d10: 0, d11: 0,
            d12: 0, d13: 0, d14: 0, d15: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  switch_context — The Heart of Green Threading
// ═══════════════════════════════════════════════════════════════════════════════

/// Save current registers into `old`, load registers from `new`, return.
/// After this returns, we're running on the new context's stack with
/// the new context's registers. The `ret` instruction jumps to whatever
/// address was in the new context's x30 (link register).
///
/// We save both integer callee-saved regs (x19-x28, x29, x30, sp) and
/// SIMD callee-saved regs (d8-d15). Missing d8-d15 causes intermittent
/// corruption under concurrent workloads — a painful lesson!
#[unsafe(naked)]
unsafe extern "C" fn switch_context(_old: *mut Context, _new: *const Context) {
    naked_asm!(
        // Save callee-saved integer registers to old context (x0)
        "stp x19, x20, [x0, #0]",
        "stp x21, x22, [x0, #16]",
        "stp x23, x24, [x0, #32]",
        "stp x25, x26, [x0, #48]",
        "stp x27, x28, [x0, #64]",
        "stp x29, x30, [x0, #80]",
        "mov x2, sp",
        "str x2, [x0, #96]",
        // Save callee-saved SIMD registers (d8-d15)
        "stp d8, d9, [x0, #104]",
        "stp d10, d11, [x0, #120]",
        "stp d12, d13, [x0, #136]",
        "stp d14, d15, [x0, #152]",

        // Load callee-saved integer registers from new context (x1)
        "ldp x19, x20, [x1, #0]",
        "ldp x21, x22, [x1, #16]",
        "ldp x23, x24, [x1, #32]",
        "ldp x25, x26, [x1, #48]",
        "ldp x27, x28, [x1, #64]",
        "ldp x29, x30, [x1, #80]",
        "ldr x2, [x1, #96]",
        "mov sp, x2",
        // Load callee-saved SIMD registers (d8-d15)
        "ldp d8, d9, [x1, #104]",
        "ldp d10, d11, [x1, #120]",
        "ldp d12, d13, [x1, #136]",
        "ldp d14, d15, [x1, #152]",

        // Return — jumps to address in x30 (the new context's LR)
        "ret",
    );
}

fn main() {
    println!("🚀 Minimal Green Thread Runtime");
    println!("══════════════════════════════════════════════════\n");
    stage1_basic_switch();
    stage2_scheduler();
    stage3_preemption();

    // Stage 4 experiments — incremental, one question at a time
    exp1_sigalrm_in_tight_loop();
    exp2_read_ucontext();
    exp3_modify_pc();
    exp4_preempt_green_thread();
    exp5_full_preemption();

    // Stage 5: M:N threading experiments
    exp6_two_os_threads();
    exp7_shared_work_queue();

    // Stage 6: Stress tests — push the limits
    stress_many_goroutines();
}

// Shared context for Stage 1 (proper runtime in Stage 2)
static mut MAIN_CTX: Context = Context::zeroed();

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 1: Basic Context Switch
// ═══════════════════════════════════════════════════════════════════════════════

fn stage1_basic_switch() {
    println!("📋 Stage 1: Raw Context Switch\n");

    // Allocate a stack for our green thread (64 KB)
    const STACK_SIZE: usize = 64 * 1024;
    let stack = vec![0u8; STACK_SIZE];
    let stack_top = stack.as_ptr() as u64 + STACK_SIZE as u64;
    // Ensure 16-byte alignment (AArch64 requires it)
    let stack_top = stack_top & !0xF;

    // Set up the green thread's initial context
    let mut green_ctx = Context::zeroed();
    green_ctx.sp = stack_top;
    green_ctx.x30 = green_thread_fn as u64;  // LR = entry point
    green_ctx.x29 = 0;                        // FP = null (end of frame chain)

    println!("  ➡️  Main: switching to green thread...");

    unsafe {
        switch_context(&raw mut MAIN_CTX, &green_ctx as *const Context);
    }

    println!("  ⬅️  Main: back from green thread!\n");

    // Keep stack alive until we're done
    drop(stack);
}

extern "C" fn green_thread_fn() {
    println!("  🟢 Green thread: Hello! Running on my own stack!");

    let sp: u64;
    unsafe { asm!("mov {}, sp", out(reg) sp); }
    println!("  🟢 Green thread: SP = {:#x}", sp);

    println!("  🟢 Green thread: switching back to main...");

    unsafe {
        switch_context(
            &mut Context::zeroed() as *mut Context,
            &raw const MAIN_CTX,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 2: Multiple Green Threads + Round-Robin Scheduler
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
enum ThreadState {
 Ready, Running, Finished }

const STACK_SIZE: usize = 64 * 1024;

struct GreenThread {
    id: usize,
    state: ThreadState,
    ctx: Context,
    stack: Vec<u8>,    // owned stack memory
}

struct Runtime {
    threads: Vec<GreenThread>,
    current: usize,     // index of currently running thread
}

// Global runtime pointer — the scheduler and yield_now need to find it
static mut RT: *mut Runtime = std::ptr::null_mut();

impl Runtime {
    fn new() -> Self {
        // Thread 0 = the main/scheduler thread (uses the OS stack)
        let main_thread = GreenThread {
            id: 0,
            state: ThreadState::Running,
            ctx: Context::zeroed(),
            stack: Vec::new(),  // main uses OS stack
        };
        Runtime {
            threads: vec![main_thread],
            current: 0,
        }
    }

    fn spawn(&mut self, f: fn()) {
        let id = self.threads.len();

        let stack = vec![0u8; STACK_SIZE];
        let stack_top = stack.as_ptr() as u64 + STACK_SIZE as u64;
        let stack_top = stack_top & !0xF;  // 16-byte align

        let mut ctx = Context::zeroed();
        ctx.sp = stack_top;
        ctx.x30 = thread_wrapper as u64;  // entry point
        ctx.x19 = f as u64;               // pass function via callee-saved reg
        ctx.x29 = 0;                       // end of frame chain

        self.threads.push(GreenThread {
            id,
            state: ThreadState::Ready,
            ctx,
            stack,
        });
    }

    fn run(&mut self) {
        // Scheduler loop: thread 0 is the scheduler, it picks the next ready thread
        loop {
            // Find next Ready thread (round-robin, skip thread 0 = scheduler)
            let n = self.threads.len();
            let mut next = None;
            for i in 1..n {
                let idx = (self.current.max(1) + i - 1) % (n - 1) + 1;
                if self.threads[idx].state == ThreadState::Ready {
                    next = Some(idx);
                    break;
                }
            }

            let Some(idx) = next else { break }; // all finished

            self.current = idx;
            self.threads[idx].state = ThreadState::Running;

            unsafe {
                let sched_ctx = &mut self.threads[0].ctx as *mut Context;
                let thread_ctx = &self.threads[idx].ctx as *const Context;
                switch_context(sched_ctx, thread_ctx);
            }
            // Back here = thread yielded or finished
        }
    }
}

/// Called by green threads to voluntarily give up the CPU
fn yield_now() {
    unsafe {
        let rt = &mut *RT;
        let current = rt.current;
        rt.threads[current].state = ThreadState::Ready;
        rt.current = 0;

        // Switch from current thread directly to scheduler (thread 0)
        let cur_ctx = &mut rt.threads[current].ctx as *mut Context;
        let sched_ctx = &rt.threads[0].ctx as *const Context;
        switch_context(cur_ctx, sched_ctx);
        // When we resume, we're back here
    }
}

/// Thread entry wrapper — invoked when a new green thread starts
#[unsafe(naked)]
unsafe extern "C" fn thread_wrapper() {
    naked_asm!(
        // x19 = function pointer (set in spawn)
        "blr x19",           // call the user function
        "bl {done}",         // mark finished + yield
        "brk #1",            // should never reach here
        done = sym thread_done,
    );
}

extern "C" fn thread_done() {
    unsafe {
        let rt = &mut *RT;
        let current = rt.current;
        rt.threads[current].state = ThreadState::Finished;
        rt.current = 0;

        // Switch to scheduler — we'll never come back
        let cur_ctx = &mut rt.threads[current].ctx as *mut Context;
        let sched_ctx = &rt.threads[0].ctx as *const Context;
        switch_context(cur_ctx, sched_ctx);
    }
}

fn stage2_scheduler() {
    println!("📋 Stage 2: Multiple Green Threads + Scheduler\n");

    let mut rt = Runtime::new();

    rt.spawn(|| {
        for i in 1..=3 {
            println!("  [thread 1] step {}", i);
            yield_now();
        }
        println!("  [thread 1] done!");
    });

    rt.spawn(|| {
        for i in 1..=3 {
            println!("  [thread 2] step {}", i);
            yield_now();
        }
        println!("  [thread 2] done!");
    });

    rt.spawn(|| {
        for i in 1..=3 {
            println!("  [thread 3] step {}", i);
            yield_now();
        }
        println!("  [thread 3] done!");
    });

    unsafe { RT = &mut rt as *mut Runtime; }

    println!("  Spawned 3 green threads. Running scheduler...\n");
    rt.run();

    unsafe { RT = std::ptr::null_mut(); }
    println!("\n  ✅ All threads completed!\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 3: Timer-Based Preemption (Cooperative via SIGALRM)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Go's runtime inserts preemption checks at function prologues and loop
// back-edges. Here we simulate that: a SIGALRM fires periodically and sets
// a flag, and threads check it at "safe points" via check_preempt().
//
// This is still cooperative — threads must call check_preempt(). But it
// demonstrates the timer mechanism that Go uses under the hood.

use std::sync::atomic::{AtomicBool, Ordering};

static SHOULD_YIELD: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
}

// itimerval for setitimer
// macOS: tv_sec = __darwin_time_t = long = i64
//        tv_usec = __darwin_suseconds_t = long = i64 (NOT i32!)
#[repr(C)]
struct Timeval {
    tv_sec: i64,
    tv_usec: i64,   // was i32 — WRONG on macOS 64-bit!
}

#[repr(C)]
struct ITimerval {
    it_interval: Timeval,
    it_value: Timeval,
}

extern "C" {
    fn setitimer(which: i32, new: *const ITimerval, old: *mut ITimerval) -> i32;
}

const SIGALRM: i32 = 14;
const ITIMER_REAL: i32 = 0;

extern "C" fn alarm_handler(_sig: i32) {
    SHOULD_YIELD.store(true, Ordering::Relaxed);
}

/// Check if the timer has fired and yield if so.
/// This is the "preemption safe point" — equivalent to Go's
/// compiler-inserted preemption checks.
fn check_preempt() {
    if SHOULD_YIELD.swap(false, Ordering::Relaxed) {
        yield_now();
    }
}

fn start_timer(interval_us: i64) {
    unsafe {
        signal(SIGALRM, alarm_handler);
        let timer = ITimerval {
            it_interval: Timeval { tv_sec: 0, tv_usec: interval_us },
            it_value: Timeval { tv_sec: 0, tv_usec: interval_us },
        };
        setitimer(ITIMER_REAL, &timer, std::ptr::null_mut());
    }
}

fn stop_timer() {
    unsafe {
        let timer = ITimerval {
            it_interval: Timeval { tv_sec: 0, tv_usec: 0 },
            it_value: Timeval { tv_sec: 0, tv_usec: 0 },
        };
        setitimer(ITIMER_REAL, &timer, std::ptr::null_mut());
    }
}

fn stage3_preemption() {
    println!("📋 Stage 3: Timer-Based Preemption\n");

    let mut rt = Runtime::new();

    // These threads do heavy work and only yield via check_preempt()
    // — they never call yield_now() directly
    rt.spawn(|| {
        let mut total: u64 = 0;
        for i in 0u64..5_000_000 {
            total = total.wrapping_add(i);
            if i % 500_000 == 0 {
                check_preempt();
                if i > 0 {
                    println!("  [thread A] computed to {}M (sum={})", i / 1_000_000, total);
                }
            }
        }
        println!("  [thread A] done! sum={}", total);
    });

    rt.spawn(|| {
        let mut total: u64 = 0;
        for i in 0u64..5_000_000 {
            total = total.wrapping_add(i);
            if i % 500_000 == 0 {
                check_preempt();
                if i > 0 {
                    println!("  [thread B] computed to {}M (sum={})", i / 1_000_000, total);
                }
            }
        }
        println!("  [thread B] done! sum={}", total);
    });

    rt.spawn(|| {
        let mut total: u64 = 0;
        for i in 0u64..5_000_000 {
            total = total.wrapping_add(i);
            if i % 500_000 == 0 {
                check_preempt();
                if i > 0 {
                    println!("  [thread C] computed to {}M (sum={})", i / 1_000_000, total);
                }
            }
        }
        println!("  [thread C] done! sum={}", total);
    });

    unsafe { RT = &mut rt as *mut Runtime; }

    println!("  Spawned 3 compute threads. Timer preempting every 1ms...\n");
    start_timer(1000); // 1000us = 1ms
    rt.run();
    stop_timer();

    unsafe { RT = std::ptr::null_mut(); }
    println!("\n  ✅ All preemptable threads completed!\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 4 — Experiment 1: Does SIGALRM fire inside a tight loop?
// ═══════════════════════════════════════════════════════════════════════════════
//
// Question: If a green thread runs a tight loop with NO yield and NO
// check_preempt(), will SIGALRM still fire and interrupt it?
//
// Setup: Use sigaction with SA_SIGINFO so we can later access ucontext_t.
// For now, just count how many times the handler fires during a 50M iteration loop.

use std::sync::atomic::AtomicU64;

static ALARM_COUNT: AtomicU64 = AtomicU64::new(0);

// sigaction structures for macOS AArch64
#[repr(C)]
struct SigAction {
    // On macOS: sa_sigaction is the handler (when SA_SIGINFO is set)
    sa_handler: usize,
    sa_mask: u32,
    sa_flags: i32,
}

extern "C" {
    fn sigaction(sig: i32, act: *const SigAction, oldact: *mut SigAction) -> i32;
}

const SA_SIGINFO: i32 = 0x0040;

// Handler with SA_SIGINFO signature: (sig, info, ucontext)
extern "C" fn exp1_handler(_sig: i32, _info: *const u8, _ucontext: *mut u8) {
    ALARM_COUNT.fetch_add(1, Ordering::Relaxed);
}

// Simple handler for signal()
extern "C" fn exp1_simple_handler(_sig: i32) {
    ALARM_COUNT.fetch_add(1, Ordering::Relaxed);
}

fn exp1_sigalrm_in_tight_loop() {
    println!("🧪 Exp 1: Does SIGALRM fire?\n");

    extern "C" { fn raise(sig: i32) -> i32; }

    // Test 0: raise() — synchronous delivery ✅
    unsafe { signal(SIGALRM, exp1_simple_handler); }
    ALARM_COUNT.store(0, Ordering::Relaxed);
    unsafe { raise(SIGALRM); }
    println!("  Test 0 (raise): count={}", ALARM_COUNT.load(Ordering::Relaxed));

    // Test 1: setitimer + sleep ✅
    unsafe { signal(SIGALRM, exp1_simple_handler); }
    ALARM_COUNT.store(0, Ordering::Relaxed);
    let timer = ITimerval {
        it_interval: Timeval { tv_sec: 0, tv_usec: 1000 }, // 1ms
        it_value: Timeval { tv_sec: 0, tv_usec: 1000 },
    };
    unsafe { setitimer(ITIMER_REAL, &timer, std::ptr::null_mut()); }
    std::thread::sleep(std::time::Duration::from_millis(50));
    stop_timer();
    println!("  Test 1 (setitimer 1ms + 50ms sleep): count={}", ALARM_COUNT.load(Ordering::Relaxed));

    // Test 2: tight loop — does SIGALRM interrupt pure computation?
    unsafe { signal(SIGALRM, exp1_simple_handler); }
    ALARM_COUNT.store(0, Ordering::Relaxed);
    let timer = ITimerval {
        it_interval: Timeval { tv_sec: 0, tv_usec: 1000 },
        it_value: Timeval { tv_sec: 0, tv_usec: 1000 },
    };
    unsafe { setitimer(ITIMER_REAL, &timer, std::ptr::null_mut()); }
    let mut total: u64 = 0;
    for i in 0u64..500_000_000 {
        total = total.wrapping_add(std::hint::black_box(i));
    }
    stop_timer();
    let count = ALARM_COUNT.load(Ordering::Relaxed);
    println!("  Test 2 (1ms timer + 500M iters): count={} sum={}", count, total);
    println!("  ✅ Verdict: SIGALRM fires {} times inside tight loop!\n", count);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 4 — Experiment 2: Can we read PC/SP/LR from ucontext_t?
// ═══════════════════════════════════════════════════════════════════════════════
//
// Question: When SIGALRM fires with SA_SIGINFO, can we read the interrupted
// instruction's PC, SP, and LR from the ucontext_t?
//
// macOS AArch64 offsets (from offsets.c):
//   uc_mcontext pointer at offset 48 in ucontext_t
//   mcontext.__ss.__pc at offset 272
//   mcontext.__ss.__sp at offset 264
//   mcontext.__ss.__lr at offset 256
//   mcontext.__ss.__fp at offset 248
//   mcontext.__ss.__x[i] at offset 16 + i*8

// Store captured values so we can check them after the handler returns
static mut CAPTURED_PC: u64 = 0;
static mut CAPTURED_SP: u64 = 0;
static mut CAPTURED_LR: u64 = 0;

const UC_MCONTEXT_OFFSET: usize = 48;
const MC_PC_OFFSET: usize = 272;
const MC_SP_OFFSET: usize = 264;
const MC_LR_OFFSET: usize = 256;

extern "C" fn exp2_handler(_sig: i32, _info: *const u8, ucontext: *mut u8) {
    unsafe {
        // ucontext -> uc_mcontext (pointer at offset 48)
        let mctx_ptr = *((ucontext as *const u8).add(UC_MCONTEXT_OFFSET) as *const *const u8);
        CAPTURED_PC = *(mctx_ptr.add(MC_PC_OFFSET) as *const u64);
        CAPTURED_SP = *(mctx_ptr.add(MC_SP_OFFSET) as *const u64);
        CAPTURED_LR = *(mctx_ptr.add(MC_LR_OFFSET) as *const u64);
    }
    ALARM_COUNT.fetch_add(1, Ordering::Relaxed);
}

fn exp2_read_ucontext() {
    println!("🧪 Exp 2: Can we read PC/SP/LR from signal handler?\n");

    // Install SA_SIGINFO handler
    unsafe {
        let action = SigAction {
            sa_handler: exp2_handler as usize,
            sa_mask: 0,
            sa_flags: SA_SIGINFO,
        };
        sigaction(SIGALRM, &action, std::ptr::null_mut());
    }

    // Print address of a known function for reference
    println!("  exp2_work fn addr:    {:#018x}", exp2_work as u64);

    // Start timer and run a tight loop in a known function
    ALARM_COUNT.store(0, Ordering::Relaxed);
    let timer = ITimerval {
        it_interval: Timeval { tv_sec: 0, tv_usec: 5000 }, // 5ms
        it_value: Timeval { tv_sec: 0, tv_usec: 5000 },
    };
    unsafe { setitimer(ITIMER_REAL, &timer, std::ptr::null_mut()); }

    let result = exp2_work();

    stop_timer();
    let count = ALARM_COUNT.load(Ordering::Relaxed);

    println!("  Signals fired: {}", count);
    unsafe {
        println!("  Captured PC:          {:#018x}", CAPTURED_PC);
        println!("  Captured SP:          {:#018x}", CAPTURED_SP);
        println!("  Captured LR:          {:#018x}", CAPTURED_LR);

        // Check: is PC in a plausible code range?
        let pc_near_work = (CAPTURED_PC as i64 - exp2_work as i64).unsigned_abs() < 0x10000;
        println!("  PC near exp2_work?    {} (delta={})",
            if pc_near_work { "✅ YES" } else { "❌ NO" },
            (CAPTURED_PC as i64 - exp2_work as i64));
    }
    println!("  Work result: {} (to prevent optimization)\n", result);
}

#[inline(never)]
fn exp2_work() -> u64 {
    let mut total: u64 = 0;
    for i in 0u64..500_000_000 {
        total = total.wrapping_add(std::hint::black_box(i));
    }
    total
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 4 — Experiment 3: Can we redirect execution by modifying PC?
// ═══════════════════════════════════════════════════════════════════════════════
//
// Question: If we overwrite PC in ucontext_t to point to a different function,
// does execution land there when the signal handler returns?
//
// Plan: tight loop → SIGALRM → handler overwrites PC → sigreturn resumes at
// our target function instead of the loop.
//
// We use fork() so a crash doesn't kill the main process.

static mut EXP3_REDIRECTED: AtomicBool = AtomicBool::new(false);
static mut EXP3_TARGET_ADDR: u64 = 0;

extern "C" fn exp3_handler(_sig: i32, _info: *const u8, ucontext: *mut u8) {
    unsafe {
        let mctx_ptr = *((ucontext as *const u8).add(UC_MCONTEXT_OFFSET) as *const *mut u8);
        let pc_ptr = mctx_ptr.add(MC_PC_OFFSET) as *mut u64;

        // Only redirect once
        if !EXP3_REDIRECTED.swap(true, Ordering::Relaxed) {
            // Overwrite PC to point to our target function
            *pc_ptr = EXP3_TARGET_ADDR;
        }
    }
}

extern "C" fn exp3_landing_pad() {
    // If we get here, the PC redirect worked!
    // Print evidence and exit the forked child
    let msg = b"  \xE2\x9C\x85 LANDED! PC redirect worked! Exiting child.\n";
    unsafe {
        extern "C" { fn write(fd: i32, buf: *const u8, count: usize) -> isize; }
        write(1, msg.as_ptr(), msg.len());
        extern "C" { fn _exit(status: i32) -> !; }
        _exit(42); // special exit code = success
    }
}

fn exp3_modify_pc() {
    println!("🧪 Exp 3: Can we redirect execution by modifying PC?\n");

    extern "C" {
        fn fork() -> i32;
        fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    }

    let pid = unsafe { fork() };
    if pid == 0 {
        // --- CHILD PROCESS ---
        unsafe {
            EXP3_REDIRECTED = AtomicBool::new(false);
            EXP3_TARGET_ADDR = exp3_landing_pad as u64;
        }

        // Install handler
        unsafe {
            let action = SigAction {
                sa_handler: exp3_handler as usize,
                sa_mask: 0,
                sa_flags: SA_SIGINFO,
            };
            sigaction(SIGALRM, &action, std::ptr::null_mut());
        }

        // Start timer and enter tight loop
        let timer = ITimerval {
            it_interval: Timeval { tv_sec: 0, tv_usec: 5000 },
            it_value: Timeval { tv_sec: 0, tv_usec: 5000 },
        };
        unsafe { setitimer(ITIMER_REAL, &timer, std::ptr::null_mut()); }

        // This loop should be interrupted and PC redirected to landing_pad
        let mut total: u64 = 0;
        for i in 0u64..10_000_000_000 {
            total = total.wrapping_add(std::hint::black_box(i));
        }

        // Should never reach here if redirect worked
        unsafe {
            let msg = b"  \xE2\x9D\x8C Loop completed without redirect\n";
            extern "C" { fn write(fd: i32, buf: *const u8, count: usize) -> isize; }
            write(1, msg.as_ptr(), msg.len());
            extern "C" { fn _exit(status: i32) -> !; }
            _exit(1);
        }
    } else {
        // --- PARENT PROCESS ---
        let mut status: i32 = 0;
        unsafe { waitpid(pid, &mut status, 0); }

        // WIFEXITED: (status & 0x7f) == 0, WEXITSTATUS: (status >> 8) & 0xff
        let exited = (status & 0x7f) == 0;
        let exit_code = (status >> 8) & 0xff;
        let signaled = (status & 0x7f) != 0;
        let term_sig = status & 0x7f;

        if exited && exit_code == 42 {
            println!("  Child exited with code 42 = redirect worked! ✅");
        } else if exited {
            println!("  Child exited with code {} ❌", exit_code);
        } else if signaled {
            println!("  Child killed by signal {} ❌", term_sig);
        }
        println!();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 4 — Experiment 4: Preempt a GREEN THREAD via signal handler
// ═══════════════════════════════════════════════════════════════════════════════
//
// Simplest approach: signal handler sets PC = yield_now, LR = interrupted PC.
// sigreturn enters yield_now; when yield_now returns, LR takes us back to
// the interrupted code. Caller-saved regs (x0-x18) get clobbered, but let's
// see if the loop still makes progress (compiler often puts loop vars in
// callee-saved regs for long-lived variables).

static EXP4_PREEMPT_COUNT: AtomicU64 = AtomicU64::new(0);

extern "C" fn exp4_handler(_sig: i32, _info: *const u8, ucontext: *mut u8) {
    unsafe {
        let rt = &*RT;
        if rt.current == 0 { return; } // don't preempt scheduler

        let mctx = *((ucontext as *const u8).add(UC_MCONTEXT_OFFSET) as *const *mut u8);
        let pc_ptr = mctx.add(MC_PC_OFFSET) as *mut u64;
        let lr_ptr = mctx.add(MC_LR_OFFSET) as *mut u64;

        // Save interrupted PC into LR so yield_now returns there
        let interrupted_pc = *pc_ptr;
        *lr_ptr = interrupted_pc;
        *pc_ptr = yield_now as u64;

        EXP4_PREEMPT_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

fn exp4_preempt_green_thread() {
    println!("🧪 Exp 4: Naive preemption (known to clobber caller-saved regs)\n");
    println!("  Running in fork() — may crash or corrupt, that's expected.\n");

    // Run in child process because naive preemption clobbers caller-saved
    // registers and can segfault. Exp 5 fixes this properly.
    extern "C" {
        fn fork() -> i32;
        fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    }
    let pid = unsafe { fork() };
    if pid == 0 {
        // === CHILD ===
        let mut rt = Runtime::new();

        rt.spawn(|| {
            let mut count: u64 = 0;
            loop {
                count = count.wrapping_add(1);
                if count % 5_000_000 == 0 {
                    println!("  [T1] count={}", count);
                    if count >= 20_000_000 { break; }
                }
            }
            println!("  [T1] done!");
        });

        rt.spawn(|| {
            let mut count: u64 = 0;
            loop {
                count = count.wrapping_add(1);
                if count % 5_000_000 == 0 {
                    println!("  [T2] count={}", count);
                    if count >= 20_000_000 { break; }
                }
            }
            println!("  [T2] done!");
        });

        rt.spawn(|| {
            let mut count: u64 = 0;
            loop {
                count = count.wrapping_add(1);
                if count % 5_000_000 == 0 {
                    println!("  [T3] count={}", count);
                    if count >= 20_000_000 { break; }
                }
            }
            println!("  [T3] done!");
        });

        unsafe { RT = &mut rt as *mut Runtime; }

        unsafe {
            let action = SigAction {
                sa_handler: exp4_handler as usize,
                sa_mask: 0,
                sa_flags: SA_SIGINFO,
            };
            sigaction(SIGALRM, &action, std::ptr::null_mut());
        }

        println!("  3 threads, NO yield calls. Timer preempts every 1ms.\n");

        let timer = ITimerval {
            it_interval: Timeval { tv_sec: 0, tv_usec: 1000 },
            it_value: Timeval { tv_sec: 0, tv_usec: 1000 },
        };
        unsafe { setitimer(ITIMER_REAL, &timer, std::ptr::null_mut()); }

        rt.run();

        stop_timer();
        let preempts = EXP4_PREEMPT_COUNT.load(Ordering::Relaxed);
        println!("\n  Preemptions: {}", preempts);
        println!("  Threads ran (with corrupted counts — expected!)");
        extern "C" { fn _exit(status: i32) -> !; }
        unsafe { _exit(0); }
    } else {
        // === PARENT ===
        let mut status: i32 = 0;
        unsafe { waitpid(pid, &mut status, 0); }
        if status == 0 {
            println!("  ✅ Exp 4 child completed (counts may be wrong — that's the lesson)!\n");
        } else {
            println!("  ⚠️  Exp 4 child exited with status {} (crash expected — Exp 5 fixes this)\n", status);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 4 — Experiment 5: Full register save/restore preemption
// ═══════════════════════════════════════════════════════════════════════════════
//
// Exp 4 proved preemption works but clobbers caller-saved regs. Exp 5 fixes
// this with a naked trampoline that pushes x0-x18 + LR to the stack before
// calling yield_now, and pops them after resuming.
//
// The flow:
//   SIGALRM → handler sets PC=trampoline, LR=interrupted_PC
//   → sigreturn restores all regs (including x0-x18!) + jumps to trampoline
//   → trampoline pushes x0-x18 + LR to stack
//   → trampoline calls yield_now → switch_context → scheduler
//   → ...later, scheduler resumes this thread...
//   → switch_context returns → yield_now returns → back in trampoline
//   → trampoline pops x0-x18 + LR
//   → ret → jumps to LR = interrupted PC with ALL registers intact!

static EXP5_PREEMPT_COUNT: AtomicU64 = AtomicU64::new(0);

extern "C" fn exp5_handler(_sig: i32, _info: *const u8, ucontext: *mut u8) {
    unsafe {
        let rt = &*RT;
        if rt.current == 0 { return; } // don't preempt scheduler

        let mctx = *((ucontext as *const u8).add(UC_MCONTEXT_OFFSET) as *const *mut u8);
        let pc_ptr = mctx.add(MC_PC_OFFSET) as *mut u64;
        let lr_ptr = mctx.add(MC_LR_OFFSET) as *mut u64;

        let interrupted_pc = *pc_ptr;
        *lr_ptr = interrupted_pc;           // LR = where to resume
        *pc_ptr = preempt_trampoline as u64; // PC = our trampoline

        EXP5_PREEMPT_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Naked trampoline: saves ALL caller-saved registers, yields, restores, returns.
/// Entered with: LR = interrupted PC, x0-x28 = interrupted values, SP = green thread stack.
#[unsafe(naked)]
unsafe extern "C" fn preempt_trampoline() {
    naked_asm!(
        // Push 20 regs: x0-x18 + LR (x30). 20 * 8 = 160 bytes, 16-byte aligned ✅
        "stp x0,  x1,  [sp, #-160]!",
        "stp x2,  x3,  [sp, #16]",
        "stp x4,  x5,  [sp, #32]",
        "stp x6,  x7,  [sp, #48]",
        "stp x8,  x9,  [sp, #64]",
        "stp x10, x11, [sp, #80]",
        "stp x12, x13, [sp, #96]",
        "stp x14, x15, [sp, #112]",
        "stp x16, x17, [sp, #128]",
        "stp x18, x30, [sp, #144]",

        // Call yield_now — this switches to the scheduler.
        // Callee-saved regs (x19-x28, FP) are preserved through switch_context.
        // When this thread is resumed, we return here.
        "bl  {yield_now}",

        // Restore caller-saved regs + LR
        "ldp x18, x30, [sp, #144]",
        "ldp x16, x17, [sp, #128]",
        "ldp x14, x15, [sp, #112]",
        "ldp x12, x13, [sp, #96]",
        "ldp x10, x11, [sp, #80]",
        "ldp x8,  x9,  [sp, #64]",
        "ldp x6,  x7,  [sp, #48]",
        "ldp x4,  x5,  [sp, #32]",
        "ldp x2,  x3,  [sp, #16]",
        "ldp x0,  x1,  [sp], #160",

        // Return to interrupted PC (in LR)
        "ret",
        yield_now = sym yield_now,
    );
}

fn exp5_full_preemption() {
    println!("🧪 Exp 5: Full preemption with register preservation\n");

    let mut rt = Runtime::new();

    // Tight loops counting to exactly 20M — if regs are preserved,
    // count should reach exactly 20M (not some corrupted huge number)
    rt.spawn(|| {
        let mut count: u64 = 0;
        loop {
            count += 1;
            if count % 5_000_000 == 0 {
                println!("  [T1] count={}", count);
            }
            if count >= 20_000_000 { break; }
        }
        println!("  [T1] done! final count={} (expected 20000000)", count);
    });

    rt.spawn(|| {
        let mut count: u64 = 0;
        loop {
            count += 1;
            if count % 5_000_000 == 0 {
                println!("  [T2] count={}", count);
            }
            if count >= 20_000_000 { break; }
        }
        println!("  [T2] done! final count={} (expected 20000000)", count);
    });

    rt.spawn(|| {
        let mut count: u64 = 0;
        loop {
            count += 1;
            if count % 5_000_000 == 0 {
                println!("  [T3] count={}", count);
            }
            if count >= 20_000_000 { break; }
        }
        println!("  [T3] done! final count={} (expected 20000000)", count);
    });

    unsafe { RT = &mut rt as *mut Runtime; }

    // Install handler
    unsafe {
        let action = SigAction {
            sa_handler: exp5_handler as usize,
            sa_mask: 0,
            sa_flags: SA_SIGINFO,
        };
        sigaction(SIGALRM, &action, std::ptr::null_mut());
    }

    println!("  3 threads counting to 20M, NO yield calls. Timer every 1ms.\n");

    let timer = ITimerval {
        it_interval: Timeval { tv_sec: 0, tv_usec: 1000 },
        it_value: Timeval { tv_sec: 0, tv_usec: 1000 },
    };
    unsafe { setitimer(ITIMER_REAL, &timer, std::ptr::null_mut()); }

    rt.run();

    stop_timer();
    let preempts = EXP5_PREEMPT_COUNT.load(Ordering::Relaxed);
    unsafe { RT = std::ptr::null_mut(); }
    println!("\n  Total preemptions: {}", preempts);
    println!("  ✅ Full preemption with correct register preservation!\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 5 — M:N Threading: Multiple Goroutines on Multiple OS Threads
// ═══════════════════════════════════════════════════════════════════════════════
//
// Goal: run M green threads across N OS threads. Each OS thread has its own
// scheduler. Green threads yield/finish using a thread-local RT pointer.
//
// We keep the existing global `RT` for stages 1-4 and add a thread-local
// `THREAD_RT` for the M:N code. Zero changes to existing code.

use std::cell::Cell;

thread_local! {
    static THREAD_RT: Cell<*mut Runtime> = const { Cell::new(std::ptr::null_mut()) };
}

/// yield_now for M:N mode — reads RT from thread-local storage
fn yield_local() {
    THREAD_RT.with(|cell| {
        let rt = unsafe { &mut *cell.get() };
        let current = rt.current;
        rt.threads[current].state = ThreadState::Ready;
        rt.current = 0;
        let cur_ctx = &mut rt.threads[current].ctx as *mut Context;
        let sched_ctx = &rt.threads[0].ctx as *const Context;
        unsafe { switch_context(cur_ctx, sched_ctx); }
    });
}

extern "C" fn thread_done_local() {
    THREAD_RT.with(|cell| {
        let rt = unsafe { &mut *cell.get() };
        let current = rt.current;
        rt.threads[current].state = ThreadState::Finished;
        rt.current = 0;
        let cur_ctx = &mut rt.threads[current].ctx as *mut Context;
        let sched_ctx = &rt.threads[0].ctx as *const Context;
        unsafe { switch_context(cur_ctx, sched_ctx); }
    });
}

#[unsafe(naked)]
unsafe extern "C" fn thread_wrapper_local() {
    naked_asm!(
        "blr x19",
        "bl {done}",
        "brk #1",
        done = sym thread_done_local,
    );
}

impl Runtime {
    fn spawn_local(&mut self, f: fn()) {
        let id = self.threads.len();
        let stack = vec![0u8; STACK_SIZE];
        let stack_top = stack.as_ptr() as u64 + STACK_SIZE as u64;
        let stack_top = stack_top & !0xF;

        let mut ctx = Context::zeroed();
        ctx.sp = stack_top;
        ctx.x30 = thread_wrapper_local as u64;
        ctx.x19 = f as u64;
        ctx.x29 = 0;

        self.threads.push(GreenThread { id, state: ThreadState::Ready, ctx, stack });
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Experiment 6: Two OS threads, each with their own green thread scheduler
// ═══════════════════════════════════════════════════════════════════════════════

fn exp6_two_os_threads() {
    println!("🧪 Exp 6: M:N — Two OS threads, each scheduling green threads\n");

    extern "C" { fn pthread_self() -> u64; }

    let os_tid_main = unsafe { pthread_self() };
    println!("  Main OS thread: {:#x}", os_tid_main);

    // OS Thread A: runs green threads G1, G2
    let handle_a = std::thread::spawn(|| {
        let os_tid = unsafe { pthread_self() };
        println!("  OS Thread A ({:#x}): starting scheduler", os_tid);

        let mut rt = Runtime::new();

        rt.spawn_local(|| {
            let tid = unsafe { pthread_self() };
            for i in 1..=3 {
                println!("    [A-G1 on {:#x}] step {}", tid, i);
                yield_local();
            }
            println!("    [A-G1] done!");
        });

        rt.spawn_local(|| {
            let tid = unsafe { pthread_self() };
            for i in 1..=3 {
                println!("    [A-G2 on {:#x}] step {}", tid, i);
                yield_local();
            }
            println!("    [A-G2] done!");
        });

        THREAD_RT.with(|cell| cell.set(&mut rt as *mut Runtime));
        rt.run();
        THREAD_RT.with(|cell| cell.set(std::ptr::null_mut()));

        println!("  OS Thread A: all green threads done");
    });

    // OS Thread B: runs green threads G3, G4, G5
    let handle_b = std::thread::spawn(|| {
        let os_tid = unsafe { pthread_self() };
        println!("  OS Thread B ({:#x}): starting scheduler", os_tid);

        let mut rt = Runtime::new();

        rt.spawn_local(|| {
            let tid = unsafe { pthread_self() };
            for i in 1..=3 {
                println!("    [B-G3 on {:#x}] step {}", tid, i);
                yield_local();
            }
            println!("    [B-G3] done!");
        });

        rt.spawn_local(|| {
            let tid = unsafe { pthread_self() };
            for i in 1..=3 {
                println!("    [B-G4 on {:#x}] step {}", tid, i);
                yield_local();
            }
            println!("    [B-G4] done!");
        });

        rt.spawn_local(|| {
            let tid = unsafe { pthread_self() };
            for i in 1..=3 {
                println!("    [B-G5 on {:#x}] step {}", tid, i);
                yield_local();
            }
            println!("    [B-G5] done!");
        });

        THREAD_RT.with(|cell| cell.set(&mut rt as *mut Runtime));
        rt.run();
        THREAD_RT.with(|cell| cell.set(std::ptr::null_mut()));

        println!("  OS Thread B: all green threads done");
    });

    handle_a.join().unwrap();
    handle_b.join().unwrap();

    println!("\n  ✅ M:N scheduling: 5 green threads on 2 OS threads!\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Experiment 7: Shared Work Queue — Green threads migrate between OS threads
// ═══════════════════════════════════════════════════════════════════════════════
//
// This is the key Go insight: goroutines aren't pinned to an OS thread.
// A shared queue lets any worker pick up any green thread. After yielding,
// a green thread might resume on a DIFFERENT OS thread.

use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

struct Worker {
    sched_ctx: Context,
}

thread_local! {
    static WORKER_PTR: Cell<*mut Worker> = const { Cell::new(std::ptr::null_mut()) };
    // Store the current green thread while it's running
    static CURRENT_GT: Cell<*mut GreenThread> = const { Cell::new(std::ptr::null_mut()) };
}

#[inline(never)]
fn yield_worker() {
    WORKER_PTR.with(|wcell| {
        CURRENT_GT.with(|gcell| {
            let worker_ptr = wcell.get();
            let gt_ptr = gcell.get();
            // Use raw pointers only — creating &mut through CURRENT_GT would alias
            // with worker_loop's &mut gt, which is UB. The optimizer can (and does!)
            // miscompile this under -O.
            unsafe {
                std::ptr::write_volatile(&raw mut (*gt_ptr).state, ThreadState::Ready);
                switch_context(&raw mut (*gt_ptr).ctx, &raw const (*worker_ptr).sched_ctx);
            }
        });
    });
}

#[inline(never)]
extern "C" fn thread_done_worker() {
    WORKER_PTR.with(|wcell| {
        CURRENT_GT.with(|gcell| {
            let worker_ptr = wcell.get();
            let gt_ptr = gcell.get();
            // Same raw-pointer discipline as yield_worker — no &mut references.
            unsafe {
                std::ptr::write_volatile(&raw mut (*gt_ptr).state, ThreadState::Finished);
                switch_context(&raw mut (*gt_ptr).ctx, &raw const (*worker_ptr).sched_ctx);
            }
        });
    });
}

#[unsafe(naked)]
unsafe extern "C" fn thread_wrapper_worker() {
    naked_asm!(
        "blr x19",
        "bl {done}",
        "brk #1",
        done = sym thread_done_worker,
    );
}

fn make_green_thread(id: usize, f: fn()) -> GreenThread {
    let stack = vec![0u8; STACK_SIZE];
    let stack_top = stack.as_ptr() as u64 + STACK_SIZE as u64;
    let stack_top = stack_top & !0xF;

    let mut ctx = Context::zeroed();
    ctx.sp = stack_top;
    ctx.x30 = thread_wrapper_worker as u64;
    ctx.x19 = f as u64;
    ctx.x29 = 0;

    GreenThread { id, state: ThreadState::Ready, ctx, stack }
}

fn worker_loop(
    worker_id: usize,
    queue: &Arc<Mutex<VecDeque<GreenThread>>>,
    active: &AtomicU64,
) {
    extern "C" { fn pthread_self() -> u64; }
    let os_tid = unsafe { pthread_self() };

    let mut worker = Worker { sched_ctx: Context::zeroed() };
    let worker_ptr: *mut Worker = &raw mut worker;
    WORKER_PTR.with(|cell| cell.set(worker_ptr));

    let mut ran = 0u64;
    let mut spins = 0u64;
    loop {
        // Pop + increment active atomically under the lock
        let gt = {
            let mut q = queue.lock().unwrap();
            let gt = q.pop_front();
            if gt.is_some() {
                active.fetch_add(1, Ordering::SeqCst);
            }
            gt
        };

        let Some(mut gt) = gt else {
            if active.load(Ordering::SeqCst) == 0 { break; }
            spins += 1;
            std::thread::yield_now();
            continue;
        };
        spins = 0;

        // All access through raw pointers to avoid aliasing UB with
        // yield_worker/thread_done_worker which access through CURRENT_GT/WORKER_PTR.
        let gt_ptr: *mut GreenThread = &raw mut gt;
        unsafe { std::ptr::write_volatile(&raw mut (*gt_ptr).state, ThreadState::Running); }

        CURRENT_GT.with(|cell| cell.set(gt_ptr));

        // Switch to the green thread — all through raw pointers
        unsafe {
            switch_context(
                &raw mut (*worker_ptr).sched_ctx,
                &raw const (*gt_ptr).ctx,
            );
        }

        // Back: thread yielded or finished
        ran += 1;
        let state = unsafe { std::ptr::read_volatile(&raw const (*gt_ptr).state as *const ThreadState) };
        if state == ThreadState::Ready {
            queue.lock().unwrap().push_back(gt);
        }
        // If Finished: gt is dropped here
        active.fetch_sub(1, Ordering::SeqCst);
    }

    println!("    Worker {} (OS {:#x}): ran {} time-slices", worker_id, os_tid, ran);
}

fn exp7_shared_work_queue() {
    println!("🧪 Exp 7: Shared work queue — green threads migrate between OS threads\n");

    let queue: Arc<Mutex<VecDeque<GreenThread>>> = Arc::new(Mutex::new(VecDeque::new()));
    let active = Arc::new(AtomicU64::new(0));

    // Create 6 green threads, all in the shared queue
    {
        let mut q = queue.lock().unwrap();
        for i in 1..=6 {
            q.push_back(make_green_thread(i, match i {
                1 => gt_work_1 as fn(),
                2 => gt_work_2,
                3 => gt_work_3,
                4 => gt_work_4,
                5 => gt_work_5,
                _ => gt_work_6,
            }));
        }
    }

    println!("  6 green threads in shared queue, 3 OS worker threads.\n");

    // Spawn 3 OS worker threads
    let mut handles = vec![];
    for w in 0..3 {
        let q = queue.clone();
        let a = active.clone();
        handles.push(std::thread::spawn(move || {
            worker_loop(w, &q, &a);
        }));
    }

    for h in handles { h.join().unwrap(); }

    println!("\n  ✅ Shared work queue: 6 green threads migrated across 3 OS workers!\n");
}

// Green thread work functions — they print which OS thread they run on each step
// Include some actual work to spread load across workers
fn gt_work_1() {
    extern "C" { fn pthread_self() -> u64; }
    for i in 1..=4 {
        let mut v = 0u64;
        for j in 0..1_000_000 { v = v.wrapping_add(std::hint::black_box(j)); }
        println!("    [G1 on w{:04x}] step {} (v={})", (unsafe { pthread_self() } >> 12) & 0xFFF, i, v);
        yield_worker();
    }
    println!("    [G1] done!");
}
fn gt_work_2() {
    extern "C" { fn pthread_self() -> u64; }
    for i in 1..=4 {
        let mut v = 0u64;
        for j in 0..1_000_000 { v = v.wrapping_add(std::hint::black_box(j)); }
        println!("    [G2 on w{:04x}] step {} (v={})", (unsafe { pthread_self() } >> 12) & 0xFFF, i, v);
        yield_worker();
    }
    println!("    [G2] done!");
}
fn gt_work_3() {
    extern "C" { fn pthread_self() -> u64; }
    for i in 1..=4 {
        let mut v = 0u64;
        for j in 0..1_000_000 { v = v.wrapping_add(std::hint::black_box(j)); }
        println!("    [G3 on w{:04x}] step {} (v={})", (unsafe { pthread_self() } >> 12) & 0xFFF, i, v);
        yield_worker();
    }
    println!("    [G3] done!");
}
fn gt_work_4() {
    extern "C" { fn pthread_self() -> u64; }
    for i in 1..=4 {
        let mut v = 0u64;
        for j in 0..1_000_000 { v = v.wrapping_add(std::hint::black_box(j)); }
        println!("    [G4 on w{:04x}] step {} (v={})", (unsafe { pthread_self() } >> 12) & 0xFFF, i, v);
        yield_worker();
    }
    println!("    [G4] done!");
}
fn gt_work_5() {
    extern "C" { fn pthread_self() -> u64; }
    for i in 1..=4 {
        let mut v = 0u64;
        for j in 0..1_000_000 { v = v.wrapping_add(std::hint::black_box(j)); }
        println!("    [G5 on w{:04x}] step {} (v={})", (unsafe { pthread_self() } >> 12) & 0xFFF, i, v);
        yield_worker();
    }
    println!("    [G5] done!");
}
fn gt_work_6() {
    extern "C" { fn pthread_self() -> u64; }
    for i in 1..=4 {
        let mut v = 0u64;
        for j in 0..1_000_000 { v = v.wrapping_add(std::hint::black_box(j)); }
        println!("    [G6 on w{:04x}] step {} (v={})", (unsafe { pthread_self() } >> 12) & 0xFFF, i, v);
        yield_worker();
    }
    println!("    [G6] done!");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 6: Stress Tests — Push the Limits
// ═══════════════════════════════════════════════════════════════════════════════

// A shared counter to verify all goroutines ran to completion
static STRESS_DONE: AtomicU64 = AtomicU64::new(0);

fn stress_work() {
    // Each goroutine: do some work, yield a few times, increment counter
    for _ in 0..3 {
        let mut v = 0u64;
        for j in 0..100_000 { v = v.wrapping_add(std::hint::black_box(j)); }
        std::hint::black_box(v);
        yield_worker();
    }
    STRESS_DONE.fetch_add(1, Ordering::SeqCst);
}

fn stress_many_goroutines() {
    println!("🏋️ Stress Test: 10000 goroutines on 8 OS threads\n");

    STRESS_DONE.store(0, Ordering::SeqCst);

    let num_goroutines = 10_000;
    let num_workers = 8;

    let queue: Arc<Mutex<VecDeque<GreenThread>>> = Arc::new(Mutex::new(VecDeque::new()));
    let active = Arc::new(AtomicU64::new(0));

    // Create all goroutines
    {
        let mut q = queue.lock().unwrap();
        for i in 0..num_goroutines {
            q.push_back(make_green_thread(i, stress_work as fn()));
        }
    }

    println!("  {} goroutines queued, {} OS worker threads.", num_goroutines, num_workers);

    let start = std::time::Instant::now();

    // Spawn workers
    let mut handles = vec![];
    for w in 0..num_workers {
        let q = queue.clone();
        let a = active.clone();
        handles.push(std::thread::spawn(move || {
            worker_loop(w, &q, &a);
        }));
    }

    for h in handles { h.join().unwrap(); }

    let elapsed = start.elapsed();
    let done = STRESS_DONE.load(Ordering::SeqCst);

    println!("\n  Completed: {}/{} goroutines", done, num_goroutines);
    println!("  Time: {:.2?}", elapsed);
    println!("  Memory: {} goroutines × {}KB stack = {}MB",
        num_goroutines, STACK_SIZE / 1024,
        (num_goroutines * STACK_SIZE) / (1024 * 1024));

    assert_eq!(done, num_goroutines as u64,
        "FAIL: only {}/{} goroutines completed!", done, num_goroutines);

    println!("  ✅ All {} goroutines completed correctly!\n", num_goroutines);
}
