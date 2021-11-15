use std::fmt::Debug;
use tract_data::internal::*;

#[derive(Clone, Copy, Debug)]
pub enum OutputStoreSpec {
    View {
        axes: Option<(usize, usize)>,
        mr: usize,
        nr: usize,
    },
    Strides {
        row_byte_stride: isize,
        row_item_stride: isize,
        col_byte_stride: isize,
        col_item_stride: isize,
        mr: usize,
        nr: usize,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct OutputStore {
    pub(crate) ptr: *mut u8,
    pub(crate) row_byte_stride: isize,
    pub(crate) col_byte_stride: isize,
    pub(crate) row_item_stride: isize,
    pub(crate) col_item_stride: isize,
    pub(crate) panel_row_byte_stride: isize,
    pub(crate) panel_col_byte_stride: isize,
    pub(crate) item_size: usize,
    pub(crate) item_count: usize,
    pub(crate) mr: usize,
}

impl OutputStoreSpec {
    #[inline]
    pub unsafe fn wrap(self: &Self, tensor: &TensorView) -> OutputStore {
        let (mr, nr, row_item_stride, col_item_stride, row_byte_stride, col_byte_stride) =
            self.compute_strides(tensor);
        OutputStore {
            ptr: tensor.as_ptr_unchecked::<u8>() as _,
            row_byte_stride,
            col_byte_stride,
            row_item_stride,
            col_item_stride,
            panel_row_byte_stride: row_byte_stride * mr as isize,
            panel_col_byte_stride: col_byte_stride * nr as isize,
            item_size: tensor.datum_type().size_of(),
            mr,
            item_count: tensor.len(),
        }
    }

    #[inline]
    unsafe fn compute_strides(
        &self,
        tensor: &TensorView,
    ) -> (usize, usize, isize, isize, isize, isize) {
        let size_of = tensor.datum_type().size_of() as isize;
        match self {
            OutputStoreSpec::View { axes, mr, nr, .. } => {
                let (m_axis, n_axis) = if let Some(axes) = axes {
                    axes.clone()
                } else {
                    let rank = tensor.rank();
                    (rank - 2, rank - 1)
                };
                let tensor_strides = tensor.strides();
                let row_item_stride = *tensor_strides.get_unchecked(m_axis);
                let col_item_stride = *tensor_strides.get_unchecked(n_axis);
                let row_byte_stride = row_item_stride * size_of;
                let col_byte_stride = col_item_stride * size_of;
                (*mr, *nr, row_item_stride, col_item_stride, row_byte_stride, col_byte_stride)
            }
            OutputStoreSpec::Strides {
                row_byte_stride,
                col_byte_stride,
                col_item_stride,
                row_item_stride,
                mr,
                nr,
            } => (*mr, *nr, *row_item_stride, *col_item_stride, *row_byte_stride, *col_byte_stride),
        }
    }
}

impl OutputStore {
    #[inline]
    pub(super) unsafe fn tile_c(&self, down: usize, right: usize) -> OutputStoreKer {
        let (down, right) = (down as isize, right as isize);
        OutputStoreKer {
            ptr: self
                .ptr
                .offset(self.panel_row_byte_stride * down + self.panel_col_byte_stride * right)
                as *mut _,
            row_byte_stride: self.row_byte_stride,
            col_byte_stride: self.col_byte_stride,
            item_size: self.item_size,
        }
    }

    #[inline]
    pub fn item_size(&self) -> usize {
        self.item_size
    }

    #[inline]
    pub(super) unsafe fn set_from_tile(
        &self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        tile: &OutputStoreKer,
    ) {
        if self.item_size() == 1 {
            self.set_from_tile_t::<i8>(down, right, height, width, tile)
        } else {
            self.set_from_tile_t::<i32>(down, right, height, width, tile)
        }
    }

    #[inline]
    unsafe fn set_from_tile_t<T: Datum + Copy>(
        &self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        tile: &OutputStoreKer,
    ) {
        let tile = tile.ptr as *mut T;
        let dst = self.ptr.offset(
            (self.panel_row_byte_stride as usize * down
                + self.panel_col_byte_stride as usize * right) as isize,
        );
        for y in 0..height as isize {
            for x in 0..width as isize {
                let value = tile.offset(y + x * self.mr as isize);
                let dst =
                    dst.offset((y * self.row_byte_stride + x * self.col_byte_stride) as isize);
                *(dst as *mut T) = *value;
            }
        }
    }
}

#[repr(C)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct OutputStoreKer {
    pub ptr: *mut u8,
    pub row_byte_stride: isize,
    pub col_byte_stride: isize,
    pub item_size: usize,
}
