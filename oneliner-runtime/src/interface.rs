use super::{Access, Aligned, AlignedType};

#[cfg(feature = "ndarray")]
use ndarray::{ArrayView4, ArrayViewMut4};

pub type Shape4D = (usize, usize, usize, usize);

pub type Shape = Shape4D;

pub type TensorArray<T, const D1: usize, const D2: usize, const D3: usize, const D4: usize> =
    [[[[T; D4]; D3]; D2]; D1];

/// Four-dimensional tensor backed by an owned, aligned, fixed-size nested array.
pub struct Tensor4D<T, const D1: usize, const D2: usize, const D3: usize, const D4: usize> {
    storage: Aligned<AlignedType, TensorArray<T, D1, D2, D3, D4>>,
}

pub type Tensor<T, const D1: usize, const D2: usize, const D3: usize, const D4: usize> =
    Tensor4D<T, D1, D2, D3, D4>;

/// One aligned byte pool shared by a vMCU model's logical input, activations,
/// and logical output.
pub struct VmcuIoBuffer<const N: usize> {
    storage: Aligned<AlignedType, [u8; N]>,
}

impl<const N: usize> VmcuIoBuffer<N> {
    pub const LEN: usize = N;

    pub const fn new(value: u8) -> Self {
        Self {
            storage: Aligned([value; N]),
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.storage[..]
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.storage[..]
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_bytes_mut().as_mut_ptr()
    }

    pub const fn len(&self) -> usize {
        N
    }

    pub const fn is_empty(&self) -> bool {
        N == 0
    }
}

/// Borrowed immutable rank-4 tensor view into a vMCU I/O pool.
pub struct TensorView4<'a, T, const D1: usize, const D2: usize, const D3: usize, const D4: usize> {
    slice: &'a [T],
}

impl<'a, T, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    TensorView4<'a, T, D1, D2, D3, D4>
{
    pub const LEN: usize = D1 * D2 * D3 * D4;
    pub const SHAPE: Shape = (D1, D2, D3, D4);

    pub fn from_bytes(bytes: &'a [u8]) -> Self {
        let byte_len = Self::LEN * core::mem::size_of::<T>();
        assert_eq!(bytes.len(), byte_len, "tensor view byte length mismatch");
        assert_eq!(
            bytes.as_ptr().align_offset(core::mem::align_of::<T>()),
            0,
            "tensor view alignment mismatch"
        );
        // SAFETY: length and alignment were checked and the borrow of `bytes`
        // outlives the returned view.
        let slice = unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), Self::LEN) };
        Self { slice }
    }

    pub fn as_slice(&self) -> &[T] {
        self.slice
    }

    pub const fn dim(&self) -> Shape {
        Self::SHAPE
    }
}

/// Borrowed mutable rank-4 tensor view into a vMCU I/O pool.
pub struct TensorViewMut4<'a, T, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
{
    slice: &'a mut [T],
}

impl<'a, T, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    TensorViewMut4<'a, T, D1, D2, D3, D4>
{
    pub const LEN: usize = D1 * D2 * D3 * D4;
    pub const SHAPE: Shape = (D1, D2, D3, D4);

    pub fn from_bytes(bytes: &'a mut [u8]) -> Self {
        let byte_len = Self::LEN * core::mem::size_of::<T>();
        assert_eq!(bytes.len(), byte_len, "tensor view byte length mismatch");
        assert_eq!(
            bytes.as_ptr().align_offset(core::mem::align_of::<T>()),
            0,
            "tensor view alignment mismatch"
        );
        // SAFETY: length, exclusive access, and alignment were checked.
        let slice =
            unsafe { core::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<T>(), Self::LEN) };
        Self { slice }
    }

    pub fn as_slice(&self) -> &[T] {
        self.slice
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.slice
    }

    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.slice.fill(value);
    }

    pub const fn dim(&self) -> Shape {
        Self::SHAPE
    }
}

impl<T, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    Tensor4D<T, D1, D2, D3, D4>
{
    pub const SHAPE: Shape = (D1, D2, D3, D4);

    pub const LEN: usize = D1 * D2 * D3 * D4;

    pub const fn new(value: T) -> Self
    where
        T: Copy,
    {
        Self {
            storage: Aligned([[[[value; D4]; D3]; D2]; D1]),
        }
    }

    pub const fn from_array(storage: TensorArray<T, D1, D2, D3, D4>) -> Self {
        Self {
            storage: Aligned(storage),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage[..]
            .as_flattened()
            .as_flattened()
            .as_flattened()
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.storage[..]
            .as_flattened_mut()
            .as_flattened_mut()
            .as_flattened_mut()
    }

    pub fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_slice_mut().as_mut_ptr()
    }

    pub fn byte_len(&self) -> usize {
        core::mem::size_of_val(self.as_slice())
    }

    pub const fn len(&self) -> usize {
        Self::LEN
    }

    pub const fn is_empty(&self) -> bool {
        Self::LEN == 0
    }

    pub const fn dim(&self) -> Shape {
        Self::SHAPE
    }

    #[cfg(feature = "ndarray")]
    pub fn view(&self) -> ArrayView4<'_, T> {
        ArrayView4::from_shape(Self::SHAPE, self.as_slice())
            .expect("Tensor4D shape must match its storage")
    }

    #[cfg(feature = "ndarray")]
    pub fn view_mut(&mut self) -> ArrayViewMut4<'_, T> {
        ArrayViewMut4::from_shape(Self::SHAPE, self.as_slice_mut())
            .expect("Tensor4D shape must match its storage")
    }

    #[cfg(feature = "ndarray")]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.view_mut().fill(value);
    }

    #[cfg(not(feature = "ndarray"))]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.as_slice_mut().fill(value);
    }
}

/// Typed tensor inference implemented directly by generated model instances.
pub trait ModelInference {
    type InputTensor;
    type OutputTensor;

    /// Runs inference and returns an owned output tensor.
    fn run(&mut self, input: &Self::InputTensor) -> Self::OutputTensor;

    /// Creates a zero-filled input tensor with the model's element type and dimensions.
    fn create_input_tensor() -> Self::InputTensor;
}

/// In-place inference over one compact vMCU activation/I/O pool.
pub trait InPlaceModelInference {
    type IoBuffer;
    type InputViewMut<'a>
    where
        Self: 'a;
    type OutputView<'a>
    where
        Self: 'a;

    fn create_io_buffer() -> Self::IoBuffer;
    fn input_view_mut<'a>(pool: &'a mut Self::IoBuffer) -> Self::InputViewMut<'a>;
    fn run_in_place<'a>(&mut self, pool: &'a mut Self::IoBuffer) -> Self::OutputView<'a>;
}

/// Metadata exposed by a model generated with `#[model]`.
///
/// Input: no runtime input; metadata is provided as associated constants.
/// Output: model path and generated artifact metadata.
pub trait ModelSource {
    const MODEL_PATH: &'static str;
    const ARTIFACTS: ModelArtifacts;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    TooManyBindings {
        provided: usize,
        capacity: usize,
    },
    TooManyConstants {
        provided: usize,
        capacity: usize,
    },
    MissingDispatchFunction {
        ordinal: usize,
    },
    WorkgroupCountTooLarge {
        dimension: char,
        value: u32,
    },
    BufferRangeOutOfBounds {
        offset: usize,
        length: usize,
        capacity: usize,
    },
    DispatchFailed {
        status: i32,
    },
    InvalidAccess {
        access: Access,
        required: Access,
    },
    NullPointer,
}

impl core::fmt::Display for Error {
    /// Formats a runtime error for diagnostics.
    ///
    /// Input: formatter supplied by Rust formatting machinery.
    /// Output: `Ok(())` when the message was written.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooManyBindings { provided, capacity } => write!(
                f,
                "dispatch has {provided} bindings, but runtime stack storage holds {capacity}"
            ),
            Self::TooManyConstants { provided, capacity } => write!(
                f,
                "dispatch has {provided} constants, but the IREE ABI supports {capacity}"
            ),
            Self::MissingDispatchFunction { ordinal } => {
                write!(f, "IREE library does not expose dispatch ordinal {ordinal}")
            }
            Self::WorkgroupCountTooLarge { dimension, value } => write!(
                f,
                "workgroup count {dimension}={value} exceeds the IREE ABI limit"
            ),
            Self::BufferRangeOutOfBounds {
                offset,
                length,
                capacity,
            } => write!(
                f,
                "tensor range offset={offset} length={length} exceeds capacity {capacity}"
            ),
            Self::DispatchFailed { status } => {
                write!(f, "backend dispatch returned status {status}")
            }

            Self::InvalidAccess { access, required } => write!(
                f,
                "tensor access {access:?} is invalid; expected {required:?}"
            ),
            Self::NullPointer => write!(f, "tensor pointer is null"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelArtifacts {
    pub backend: &'static str,
    pub expansion: &'static str,
    pub model_path: &'static str,
    pub compile_input_path: &'static str,
    pub object_path: &'static str,
    pub link_path: &'static str,
    pub ir_path: &'static str,
    pub flow_rs_path: &'static str,
    pub metadata_json_path: &'static str,
    /// Size in bytes of the model's single input tensor.
    pub input_size: usize,
    /// Size in bytes of the model's single output tensor.
    pub output_size: usize,
    /// Aligned compact I/O pool bytes, or zero for the immutable tensor ABI.
    pub io_pool_size: usize,
    /// Byte offset of the logical input view in the I/O pool.
    pub input_offset: usize,
    /// Byte offset of the logical output view in the I/O pool.
    pub output_offset: usize,
    /// Model parameter/weight bytes placed in flash (read-only model data).
    pub params_size: usize,
    /// Machine code bytes of the compiled model placed in flash.
    pub code_size: usize,
    /// Read-only data bytes embedded in the compiled model object (lookup
    /// tables, library metadata, unwind tables) placed in flash.
    pub rodata_size: usize,
    /// Total model flash footprint: parameters plus compiled code and data.
    pub total_flash_size: usize,
    /// Transient workspace bytes held in RAM during inference.
    pub ram_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_pool_views_borrow_selected_ranges() {
        let mut pool = VmcuIoBuffer::<64>::new(0);
        {
            let mut input =
                TensorViewMut4::<i8, 1, 1, 1, 4>::from_bytes(&mut pool.as_bytes_mut()[8..12]);
            input.fill(-3);
        }
        let output = TensorView4::<i8, 1, 1, 1, 4>::from_bytes(&pool.as_bytes()[8..12]);
        assert_eq!(output.as_slice(), &[-3; 4]);
        assert_eq!(output.dim(), (1, 1, 1, 4));
    }
}
